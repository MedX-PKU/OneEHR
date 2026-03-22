"""Small graph utilities shared by graph-enhanced sequence models."""

from __future__ import annotations

import math

import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        std = 1.0 / math.sqrt(out_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(adj.float(), torch.matmul(x.float(), self.weight.float()))
        return y + self.bias.float() if self.bias is not None else y


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
    adj = adj + eye
    denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj / denom
