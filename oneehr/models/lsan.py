"""LSAN models for longitudinal self-attention on EHR sequences."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.grouped import GroupedVisitEncoder


class LSANBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        kernel_size: int = 3,
        group_indices: list[list[int]] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visit_encoder = GroupedVisitEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            group_indices=group_indices,
            dropout=dropout,
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.long_encoder = nn.TransformerEncoder(block, num_layers=num_layers)
        self.short_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pool_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        visits, _, _ = self.visit_encoder(x)
        steps = visits.size(1)
        mask = torch.arange(steps, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        long_term = self.long_encoder(visits, src_key_padding_mask=mask)
        short_term = self.short_conv(visits.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat([long_term, short_term], dim=-1)))
        fused = gate * long_term + (1.0 - gate) * short_term
        return self.dropout(fused)


class LSANModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        nhead: int = 4,
        num_layers: int = 2,
        kernel_size: int = 3,
        group_indices: list[list[int]] | None = None,
        group_names: list[str] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = LSANBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            kernel_size=kernel_size,
            group_indices=group_indices,
            dropout=dropout,
        )
        self.pool_proj = nn.Linear(hidden_dim, 1)
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
        fused = self.backbone(x, lengths)
        scores = self.pool_proj(torch.tanh(fused)).squeeze(-1)
        mask = torch.arange(fused.size(1), device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), fused).squeeze(1)
        return self.head(pooled)


class LSANTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.backbone = LSANBackbone(
            input_dim=int(kwargs["input_dim"]),
            hidden_dim=hidden_dim,
            nhead=int(kwargs.get("nhead", 4)),
            num_layers=int(kwargs.get("num_layers", 2)),
            kernel_size=int(kwargs.get("kernel_size", 3)),
            group_indices=kwargs.get("group_indices"),
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
        return self.head(self.backbone(x, lengths))
