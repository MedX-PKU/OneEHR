"""SAnD models for self-attentive EHR sequence learning."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.full((size, size), float("-inf"), device=device),
        diagonal=1,
    )


class CausalConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size - 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = y[:, : x.size(1), :]
        return self.norm(x + self.dropout(F.gelu(y)))


class SAnDBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.conv = CausalConv(d_model, kernel_size=kernel_size, dropout=dropout)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=num_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.conv(h)
        steps = h.size(1)
        key_padding_mask = torch.arange(steps, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        return self.encoder(
            h,
            mask=_causal_mask(steps, h.device),
            src_key_padding_mask=key_padding_mask,
        )


class SAnDModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        out_dim: int = 1,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        kernel_size: int = 3,
        interp_points: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.interp_points = interp_points
        self.backbone = SAnDBackbone(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model * interp_points, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.backbone(x, lengths)
        pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), self.interp_points).flatten(1)
        return self.head(pooled)


class SAnDTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        d_model = int(kwargs.get("d_model", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.backbone = SAnDBackbone(
            input_dim=int(kwargs["input_dim"]),
            d_model=d_model,
            nhead=int(kwargs.get("nhead", 4)),
            num_layers=int(kwargs.get("num_layers", 2)),
            dim_feedforward=int(kwargs.get("dim_feedforward", 256)),
            kernel_size=int(kwargs.get("kernel_size", 3)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.head = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.backbone(x, lengths))
