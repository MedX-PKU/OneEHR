"""Dipole models for bidirectional attention-based EHR prediction."""

from __future__ import annotations

import torch
from torch import nn


class DipoleBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        attention_type: str = "general",
        dropout: float = 0.1,
    ):
        super().__init__()
        if attention_type not in {"location", "general", "concat"}:
            raise ValueError(f"Unsupported attention_type={attention_type!r}")
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.location_proj = nn.Linear(hidden_dim * 2, 1)
        self.general_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.concat_proj = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.concat_score = nn.Linear(hidden_dim * 2, 1)
        self.fusion = nn.Linear(hidden_dim * 4, hidden_dim * 2)

    def _attn_scores(self, out: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if self.attention_type == "location":
            return self.location_proj(out).squeeze(-1)
        if self.attention_type == "general":
            q = self.general_proj(query).unsqueeze(-1)
            return torch.bmm(out, q).squeeze(-1)
        q = query.unsqueeze(1).expand_as(out)
        return self.concat_score(torch.tanh(self.concat_proj(torch.cat([out, q], dim=-1)))).squeeze(-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        query = out[torch.arange(out.size(0), device=out.device), (lengths - 1).clamp_min(0)]
        scores = self._attn_scores(out, query)
        mask = torch.arange(out.size(1), device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        return self.dropout(torch.relu(self.fusion(torch.cat([context, query], dim=-1))))


class DipoleModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        attention_type: str = "general",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = DipoleBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_type=attention_type,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.backbone(x, lengths))


class DipoleTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.core = DipoleModel(**kwargs)

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
