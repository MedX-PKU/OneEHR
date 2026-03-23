"""Raindrop-inspired graph-guided models for irregular EHR sequences."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths
from oneehr.models.time import ContinuousTimeEncoding


def _normalize_batch_adj(adj: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
    adj = adj + eye
    denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adj / denom


class RaindropBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sensor_embed = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.time_enc = ContinuousTimeEncoding(hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        missing_mask: torch.Tensor | None,
        time_delta: torch.Tensor | None,
        visit_time: torch.Tensor | None,
    ) -> torch.Tensor:
        if missing_mask is None or visit_time is None:
            raise ValueError("Raindrop requires missing_mask and visit_time in forward()")
        observed = 1.0 - missing_mask
        sensor_tokens = x.unsqueeze(-1) * self.sensor_embed.unsqueeze(0).unsqueeze(0)
        sensor_tokens = sensor_tokens * observed.unsqueeze(-1)

        presence = observed.mean(dim=1)
        sim = torch.relu(torch.matmul(self.sensor_embed, self.sensor_embed.transpose(0, 1)))
        adj = presence.unsqueeze(-1) * presence.unsqueeze(-2) * sim.unsqueeze(0)
        adj = _normalize_batch_adj(adj)
        graph_tokens = torch.einsum("bij,btjh->btih", adj, sensor_tokens)
        visit_repr = graph_tokens.mean(dim=2) + self.time_enc(visit_time)

        packed = nn.utils.rnn.pack_padded_sequence(self.dropout(visit_repr), lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.temporal_gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out


class RaindropModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = RaindropBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
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


class RaindropTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.backbone = RaindropBackbone(
            input_dim=int(kwargs["input_dim"]),
            hidden_dim=hidden_dim,
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
