"""Continuous-time helpers shared by irregular-sequence models."""

from __future__ import annotations

import math

import torch
from torch import nn


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(2, dim)

    def forward(self, visit_time: torch.Tensor, delta: torch.Tensor | None = None) -> torch.Tensor:
        if delta is None:
            delta = torch.zeros_like(visit_time)
        features = torch.stack([torch.log1p(visit_time), torch.log1p(delta)], dim=-1)
        return self.linear(features)


def pairwise_time_bias(times: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    diff = (times.unsqueeze(-1) - times.unsqueeze(-2)).abs()
    return -diff.unsqueeze(1) * scale.view(1, -1, 1, 1)


class RelativeTimeAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(num_heads) / math.sqrt(self.head_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, visit_time: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        q = self.q_proj(x).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn + pairwise_time_bias(visit_time, self.scale)
        mask = torch.arange(steps, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        weights = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, steps, self.hidden_dim)
        return self.out_proj(out)
