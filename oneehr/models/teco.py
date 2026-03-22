"""Encounter-level transformer models inspired by TECO."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths
from oneehr.models.time import ContinuousTimeEncoding


class TECOBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        static_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_enc = ContinuousTimeEncoding(hidden_dim)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=num_layers)
        self.static_proj = nn.Linear(static_dim, hidden_dim) if static_dim > 0 else None
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        visit_time: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        delta_mean = None if time_delta is None else time_delta.mean(dim=-1)
        h = self.input_proj(x)
        if visit_time is not None:
            h = h + self.time_enc(visit_time, delta_mean)

        static_token = None
        if self.static_proj is not None and static is not None:
            static_token = self.static_proj(static).unsqueeze(1)
            h = torch.cat([static_token, h], dim=1)

        prefix = 1 if static_token is not None else 0
        steps = h.size(1)
        mask = torch.arange(steps - prefix, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        if prefix:
            mask = torch.cat([torch.zeros(mask.size(0), 1, device=mask.device, dtype=mask.dtype), mask], dim=1)
        enc = self.encoder(self.norm(h), src_key_padding_mask=mask)
        if prefix:
            return enc[:, 1:, :], enc[:, 0, :]
        return enc, None


class TECOModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        static_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.backbone = TECOBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            static_dim=static_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        visit_time: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq, static_token = self.backbone(
            x,
            lengths,
            static,
            visit_time=visit_time,
            time_delta=time_delta,
        )
        pooled = static_token if static_token is not None else last_by_lengths(seq, lengths)
        return self.head(pooled)


class TECOTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.static_dim = int(kwargs.get("static_dim", 0))
        self.backbone = TECOBackbone(
            input_dim=int(kwargs["input_dim"]),
            hidden_dim=hidden_dim,
            nhead=int(kwargs.get("nhead", 4)),
            num_layers=int(kwargs.get("num_layers", 2)),
            dim_feedforward=int(kwargs.get("dim_feedforward", 256)),
            static_dim=self.static_dim,
            dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        visit_time: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq, _ = self.backbone(
            x,
            lengths,
            static,
            visit_time=visit_time,
            time_delta=time_delta,
        )
        return self.head(seq)
