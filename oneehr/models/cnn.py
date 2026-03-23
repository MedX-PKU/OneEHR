"""Temporal CNN baselines for patient-level and time-level prediction."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size - 1
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv(x)
        y = y[:, :, : x.size(-1)]
        y = self.norm(y)
        y = self.dropout(self.act(y))
        return y + residual


class CNNBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([TemporalConvBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return h.transpose(1, 2)


class CNNPatientModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = CNNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(last_by_lengths(h, lengths))


class CNNTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.backbone = CNNBackbone(
            input_dim=int(kwargs["input_dim"]),
            hidden_dim=hidden_dim,
            num_layers=int(kwargs.get("num_layers", 2)),
            kernel_size=int(kwargs.get("kernel_size", 3)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.backbone(x))
