from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _last_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    idx = (lengths - 1).clamp_min(0)
    return x[torch.arange(x.shape[0], device=x.device), idx]


class MLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTimeModel(nn.Module):
    """Per-time-step MLP head applied to each time step."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[MLPBlock(hidden_dim, dropout=dropout) for _ in range(int(num_layers))])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,T,D)
        h = self.proj(x)
        h = self.blocks(h)
        y = self.head(h)
        # mask padded steps
        t = y.shape[1]
        pad = torch.arange(t, device=lengths.device)[None, :] >= lengths[:, None]
        y = y.masked_fill(pad.unsqueeze(-1), 0.0)
        return y


class MLPModel(nn.Module):
    """Patient-level MLP via pooling last valid time step."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[MLPBlock(hidden_dim, dropout=dropout) for _ in range(int(num_layers))])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)
        h = self.blocks(h)
        last = _last_by_lengths(h, lengths)
        return self.head(last)


@dataclass(frozen=True)
class MLPArtifacts:
    feature_columns: list[str]
    state_dict: dict

