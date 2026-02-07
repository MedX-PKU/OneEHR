from __future__ import annotations

import torch
from torch import nn

from oneehr.models.utils import last_by_lengths


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
        # Support static-only tabular datasets:
        # - x: (B, T, D) for sequence-style input
        # - x: (B, D) for tabular input (no time axis)
        if x.ndim == 2:
            h = self.proj(x)
            h = self.blocks(h)
            return self.head(h)

        h = self.proj(x)
        h = self.blocks(h)
        last = last_by_lengths(h, lengths)
        return self.head(last)
