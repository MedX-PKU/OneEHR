"""M3Care model family.

Reference implementations reduce to a Transformer-style encoder with
sinusoidal positional encodings. This version keeps the interface aligned with
the rest of OneEHR and handles variable-length sequences explicitly.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class M3CareEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=num_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.positional_encoding(h)
        max_len = h.size(1)
        key_padding_mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        return self.encoder(h, src_key_padding_mask=key_padding_mask)


class M3CareModel(nn.Module):
    """Patient-level M3Care."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = M3CareEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x, lengths)
        last = last_by_lengths(h, lengths)
        return self.head(last)


class M3CareTimeModel(nn.Module):
    """Time-level M3Care via prefix rollout."""

    def __init__(self, **kwargs):
        super().__init__()
        self.core = M3CareModel(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for t in range(x.size(1)):
            cur_x = x[:, : t + 1, :]
            cur_lengths = lengths.clamp(max=t + 1).clamp(min=1)
            outputs.append(self.core(cur_x, cur_lengths))
        return torch.stack(outputs, dim=1)
