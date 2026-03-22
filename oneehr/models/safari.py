"""SAFARI model family.

This implementation keeps the reference model's MCGRU + attention flavour
while removing dataset-specific shape assumptions.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths


def resolve_safari_dim_list(*, feature_schema: list[dict] | None, input_dim: int) -> list[int]:
    if feature_schema:
        dim_list = [int(entry.get("dim", 1)) for entry in feature_schema]
        if sum(dim_list) == input_dim:
            return dim_list
    return [1] * input_dim


class FinalAttentionQKV(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.w_q(x[:, -1, :]).unsqueeze(-1)
        k = self.w_k(x)
        v = self.w_v(x)
        attn = torch.matmul(k, q).squeeze(-1) / math.sqrt(k.size(-1))
        weights = self.dropout(torch.softmax(attn, dim=-1))
        return torch.bmm(weights.unsqueeze(1), v).squeeze(1)


class SafariMCGRU(nn.Module):
    def __init__(self, dim_list: list[int], hidden_dim: int):
        super().__init__()
        self.dim_list = list(dim_list)
        self.grus = nn.ModuleList([
            nn.GRU(dim, hidden_dim, num_layers=1, batch_first=True)
            for dim in self.dim_list
        ])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs = []
        offset = 0
        for dim, gru in zip(self.dim_list, self.grus, strict=True):
            out_i, _ = gru(x[:, :, offset : offset + dim])
            outputs.append(last_by_lengths(out_i, lengths))
            offset += dim
        return torch.stack(outputs, dim=1)


class SafariBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dim_list: list[int] | None = None,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.dim_list = list(dim_list or [1] * input_dim)
        self.encoder = SafariMCGRU(self.dim_list, hidden_dim)
        self.attention = FinalAttentionQKV(hidden_dim, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        if static_dim > 0:
            self.static_proj = nn.Linear(static_dim, hidden_dim)
            self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.static_proj = None
            self.feature_proj = None

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.encoder(x, lengths)
        if self.static_proj is not None and static is not None:
            static_token = self.feature_proj(self.relu(self.static_proj(static))).unsqueeze(1)
            tokens = torch.cat([tokens, static_token], dim=1)
        context = self.attention(self.dropout(tokens))
        return self.relu(self.output_proj(self.dropout(context)))


class SafariModel(nn.Module):
    """Patient-level SAFARI."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 1,
        dim_list: list[int] | None = None,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.backbone = SafariBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dim_list=dim_list,
            static_dim=static_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.backbone(x, lengths, static))


class SafariTimeModel(nn.Module):
    """Time-level SAFARI via prefix rollout."""

    def __init__(self, **kwargs):
        super().__init__()
        self.core = SafariModel(**kwargs)
        self.static_dim = self.core.static_dim

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
            outputs.append(self.core(cur_x, cur_lengths, static))
        return torch.stack(outputs, dim=1)
