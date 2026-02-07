from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


from oneehr.models.utils import last_by_lengths


class MCGRUEncoder(nn.Module):
    """Multi-channel GRU encoder with optional static features.

    Dynamic input: x (B, T, D)
    Static input:  static (B, S)
    Output: hidden states per time step (B, T, H)
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        feat_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)
        self.feat_dim = int(feat_dim)

        self.static_proj = nn.Identity() if self.static_dim == 0 else nn.Linear(static_dim, hidden_dim)
        self.in_proj = nn.Linear(input_dim, input_dim)
        self.grus = nn.ModuleList([nn.GRU(1, feat_dim, num_layers=1, batch_first=True) for _ in range(input_dim)])
        self.out_proj = nn.Linear(input_dim * feat_dim + (hidden_dim if self.static_dim else 0), hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        bsz, t, d = x.shape
        if d != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got D={d}")

        x = self.in_proj(x)
        out = x.new_zeros((bsz, t, self.input_dim, self.feat_dim))
        for i, gru in enumerate(self.grus):
            cur = x[:, :, i].unsqueeze(-1)
            cur, _ = gru(cur)
            out[:, :, i] = cur
        out = out.flatten(2)

        if self.static_dim and static is not None:
            s = self.static_proj(static).unsqueeze(1).expand(bsz, t, -1)
            out = torch.cat([s, out], dim=-1)

        out = self.dropout(self.out_proj(out))
        return out


class MCGRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        feat_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MCGRUEncoder(
            input_dim=input_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x, static=static)
        last = last_by_lengths(z, lengths)
        return self.head(last)


class MCGRUTimeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        feat_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MCGRUEncoder(
            input_dim=input_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x, static=static)
        return self.head(z)


@dataclass(frozen=True)
class MCGRUArtifacts:
    feature_columns: list[str]
    state_dict: dict
