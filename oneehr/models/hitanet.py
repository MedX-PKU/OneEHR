"""HiTANet models for hierarchical time-aware EHR prediction."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.grouped import GroupedVisitEncoder


class TimeAwareAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(2, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        seq: torch.Tensor,
        query: torch.Tensor,
        lengths: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(seq)
        t = self.time_proj(time_features)
        scores = self.score_proj(torch.tanh(k + q + t)).squeeze(-1)
        mask = torch.arange(seq.size(1), device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), seq).squeeze(1)


class HiTANetBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
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
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.time_proj = nn.Linear(2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = TimeAwareAttention(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        visit_time: torch.Tensor | None,
        time_delta: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visit_time is None or time_delta is None:
            raise ValueError("HiTANet requires visit_time and time_delta in forward()")
        visits, _, _ = self.visit_encoder(x)
        delta_mean = torch.log1p(time_delta.mean(dim=-1))
        time_feat = torch.stack([torch.log1p(visit_time), delta_mean], dim=-1)
        packed_in = visits + self.time_proj(time_feat)
        packed = nn.utils.rnn.pack_padded_sequence(packed_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.temporal_gru(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        query = seq[torch.arange(seq.size(0), device=seq.device), (lengths - 1).clamp_min(0)]
        context = self.attention(self.dropout(seq), query, lengths, time_feat[:, : seq.size(1), :])
        return context, seq


class HiTANetModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        group_indices: list[list[int]] | None = None,
        group_names: list[str] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = HiTANetBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            group_indices=group_indices,
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
        context, _ = self.backbone(x, lengths, visit_time=visit_time, time_delta=time_delta)
        return self.head(context)


class HiTANetTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.core = HiTANetModel(**kwargs)

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
        if visit_time is None or time_delta is None:
            raise ValueError("HiTANet requires visit_time and time_delta in forward()")
        outputs = []
        for t in range(x.size(1)):
            cur_x = x[:, : t + 1, :]
            cur_lengths = lengths.clamp(max=t + 1).clamp(min=1)
            outputs.append(
                self.core(
                    cur_x,
                    cur_lengths,
                    static,
                    visit_time=visit_time[:, : t + 1],
                    time_delta=time_delta[:, : t + 1, :],
                    missing_mask=None if missing_mask is None else missing_mask[:, : t + 1, :],
                )
            )
        return torch.stack(outputs, dim=1)
