"""MCGRU (Multi-Channel GRU) model for patient-level and time-level prediction.

Per-channel GRU with static feature projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oneehr.models.recurrent import last_by_lengths


class MCGRULayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 32,
        feat_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim

        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.lab_proj = nn.Linear(input_dim, input_dim)
        self.grus = nn.ModuleList([
            nn.GRU(1, feat_dim, batch_first=True) for _ in range(input_dim)
        ])
        self.out_proj = nn.Linear(input_dim * feat_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        demo = self.static_proj(static)  # (B, H)
        x = self.lab_proj(x)

        chan_outs = []
        for i, gru in enumerate(self.grus):
            out_i, _ = gru(x[:, :, i].unsqueeze(-1))  # (B, T, feat_dim)
            chan_outs.append(out_i)
        out = torch.cat(chan_outs, dim=-1)  # (B, T, D*feat_dim)
        out = torch.cat([demo.unsqueeze(1).expand(-1, T, -1), out], dim=-1)
        return self.out_proj(self.dropout(out))  # (B, T, H)


class MCGRUModel(nn.Module):
    """Patient-level MCGRU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        feat_dim: int = 8,
        out_dim: int = 1,
        static_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.static_dim = static_dim
        if static_dim <= 0:
            raise ValueError("MCGRU requires static_dim > 0")
        self.layer = MCGRULayer(input_dim, static_dim, hidden_dim, feat_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if static is None:
            raise ValueError("MCGRU requires static features")
        h = self.layer(x, static)
        last = last_by_lengths(h, lengths)
        return self.head(last)


class MCGRUTimeModel(nn.Module):
    """Time-level MCGRU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        feat_dim: int = 8,
        out_dim: int = 1,
        static_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.static_dim = static_dim
        if static_dim <= 0:
            raise ValueError("MCGRU requires static_dim > 0")
        self.layer = MCGRULayer(input_dim, static_dim, hidden_dim, feat_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if static is None:
            raise ValueError("MCGRU requires static features")
        h = self.layer(x, static)
        return self.head(h)
