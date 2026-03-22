"""mTAND models for irregularly sampled EHR sequences."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths
from oneehr.models.time import ContinuousTimeEncoding, RelativeTimeAttention


class MTANDBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = RelativeTimeAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, visit_time: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.norm1(x), lengths, visit_time)
        x = x + self.dropout(h)
        return x + self.dropout(self.ffn(self.norm2(x)))


class MTANDBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_enc = ContinuousTimeEncoding(hidden_dim)
        self.obs_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([
            MTANDBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        missing_mask: torch.Tensor | None,
        time_delta: torch.Tensor | None,
        visit_time: torch.Tensor | None,
    ) -> torch.Tensor:
        if missing_mask is None or time_delta is None or visit_time is None:
            raise ValueError("mTAND requires missing_mask, time_delta, and visit_time in forward()")
        obs_density = 1.0 - missing_mask.mean(dim=-1, keepdim=True)
        delta_mean = time_delta.mean(dim=-1)
        h = self.input_proj(x) + self.time_enc(visit_time, delta_mean) + self.obs_proj(obs_density)
        for layer in self.layers:
            h = layer(h, lengths, visit_time)
        return h


class MTANDModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = MTANDBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        missing_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        visit_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.backbone(
            x,
            lengths,
            missing_mask=missing_mask,
            time_delta=time_delta,
            visit_time=visit_time,
        )
        return self.head(last_by_lengths(h, lengths))


class MTANDTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.backbone = MTANDBackbone(
            input_dim=int(kwargs["input_dim"]),
            hidden_dim=hidden_dim,
            num_heads=int(kwargs.get("num_heads", 4)),
            num_layers=int(kwargs.get("num_layers", 2)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        missing_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        visit_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(
            self.backbone(
                x,
                lengths,
                missing_mask=missing_mask,
                time_delta=time_delta,
                visit_time=visit_time,
            )
        )
