from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        out_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "last",
    ):
        super().__init__()
        if pooling not in {"last", "mean"}:
            raise ValueError(f"Unsupported pooling={pooling!r}")
        self.pooling = pooling

        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.in_proj(x)
        t = h.shape[1]
        # True means "ignore" for src_key_padding_mask
        key_padding_mask = torch.arange(t, device=lengths.device)[None, :] >= lengths[:, None]
        z = self.encoder(h, src_key_padding_mask=key_padding_mask)

        if self.pooling == "last":
            idx = (lengths - 1).clamp_min(0)
            pooled = z[torch.arange(z.shape[0], device=z.device), idx]
        else:
            valid = (~key_padding_mask).to(z.dtype)
            denom = valid.sum(dim=1).clamp_min(1.0)
            pooled = (z * valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)

        return self.head(pooled)


class TransformerTimeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        out_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        t = h.shape[1]
        key_padding_mask = torch.arange(t, device=lengths.device)[None, :] >= lengths[:, None]
        z = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.head(z)


@dataclass(frozen=True)
class TransformerArtifacts:
    feature_columns: list[str]
    state_dict: dict
