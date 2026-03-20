"""Deepr model: simple Conv1d + max-pool for EHR sequences."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths


class DeeprEncoder(nn.Module):
    """Linear projection → Conv1d → ReLU → Dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, window: int = 1, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        kernel_size = 2 * window + 1
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=window)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        h = self.proj(x)  # (B, T, hidden)
        h = h.transpose(1, 2)  # (B, hidden, T) for Conv1d
        h = self.act(self.conv(h))  # (B, hidden, T)
        h = h.transpose(1, 2)  # (B, T, hidden)
        return self.drop(h)


class DeeprModel(nn.Module):
    """Patient-level Deepr: encoder → max-pool over time → head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        window: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = DeeprEncoder(input_dim, hidden_dim, window, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)  # (B, T, hidden)
        pooled = h.max(dim=1).values  # (B, hidden)
        return self.head(pooled)


class DeeprTimeModel(nn.Module):
    """Time-level Deepr: encoder → head per-timestep."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        window: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = DeeprEncoder(input_dim, hidden_dim, window, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)  # (B, T, hidden)
        return self.head(h)  # (B, T, out_dim)
