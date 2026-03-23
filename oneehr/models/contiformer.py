"""Continuous-time transformer models for irregular EHR sequences."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths
from oneehr.models.time import ContinuousTimeEncoding, RelativeTimeAttention


class ContiFormerBackbone(nn.Module):
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
        self.state_decay = nn.Linear(1, hidden_dim)
        self.attn_layers = nn.ModuleList([RelativeTimeAttention(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        visit_time: torch.Tensor | None,
        time_delta: torch.Tensor | None,
        missing_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if visit_time is None or time_delta is None:
            raise ValueError("ContiFormer requires visit_time and time_delta in forward()")
        delta_mean = time_delta.mean(dim=-1, keepdim=True)
        base = self.input_proj(x) + self.time_enc(visit_time, delta_mean.squeeze(-1))

        state = torch.zeros_like(base[:, 0, :])
        states = []
        for t in range(base.size(1)):
            valid = (t < lengths).to(dtype=base.dtype, device=base.device).unsqueeze(-1)
            decay = torch.exp(-torch.relu(self.state_decay(delta_mean[:, t, :])))
            state_candidate = decay * state + base[:, t, :]
            state = valid * state_candidate + (1.0 - valid) * state
            states.append(state * valid)
        h = torch.stack(states, dim=1)

        for attn, norm in zip(self.attn_layers, self.norms, strict=True):
            h = h + self.dropout(attn(norm(h), lengths, visit_time))
        return h


class ContiFormerModel(nn.Module):
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
        self.backbone = ContiFormerBackbone(
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
        visit_time: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.backbone(
            x,
            lengths,
            visit_time=visit_time,
            time_delta=time_delta,
            missing_mask=missing_mask,
        )
        return self.head(last_by_lengths(h, lengths))


class ContiFormerTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.backbone = ContiFormerBackbone(
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
        visit_time: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(
            self.backbone(
                x,
                lengths,
                visit_time=visit_time,
                time_delta=time_delta,
                missing_mask=missing_mask,
            )
        )
