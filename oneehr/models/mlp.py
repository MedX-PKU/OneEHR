"""MLP model for patient-level and time-level prediction."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths


class MLPLayer(nn.Module):
    """Per-timestep MLP: project → hidden → 4*hidden → hidden."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.proj(x))


class MLPModel(nn.Module):
    """Patient-level MLP: applies per-timestep MLP then pools last valid step."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer = MLPLayer(input_dim, hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.layer(x)  # (B, T, hidden)
        last = last_by_lengths(h, lengths)  # (B, hidden)
        return self.head(last)


class MLPTimeModel(nn.Module):
    """Time-level MLP: applies per-timestep MLP and projects all steps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer = MLPLayer(input_dim, hidden_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.layer(x)  # (B, T, hidden)
        return self.head(h)
