"""KerPrint-style local/global KG enhanced EHR models."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.kg_backbone import KGVisitEncoder, PatientGraphSummary
from oneehr.models.recurrent import last_by_lengths
from oneehr.models.time import ContinuousTimeEncoding


class KerPrintBackbone(nn.Module):
    def __init__(self, num_groups: int, hidden_dim: int = 128, global_adj: torch.Tensor | None = None, dropout: float = 0.1):
        super().__init__()
        self.visit_encoder = KGVisitEncoder(num_groups, hidden_dim, global_adj=global_adj, dropout=dropout)
        self.graph_summary = PatientGraphSummary(hidden_dim)
        self.time_enc = ContinuousTimeEncoding(hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.knowledge_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.local_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        lengths: torch.Tensor,
        *,
        group_values: torch.Tensor | None,
        group_mask: torch.Tensor | None,
        visit_time: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if group_values is None or group_mask is None or visit_time is None:
            raise ValueError("KerPrint requires group_values, group_mask, and visit_time in forward()")
        visits, nodes, patient_mask = self.visit_encoder(group_values, group_mask)
        summary, local_summary, global_summary = self.graph_summary(nodes, patient_mask)
        visits = visits + self.time_enc(visit_time)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.dropout(visits), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.temporal_gru(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        seq = self.context_proj(seq)
        query = last_by_lengths(seq, lengths)

        recency = visit_time[:, : seq.size(1)]
        recency = recency.max(dim=1, keepdim=True).values - recency
        scores = self.score_proj(
            torch.tanh(
                self.key_proj(seq)
                + self.query_proj(query).unsqueeze(1)
                + self.time_proj(torch.log1p(recency).unsqueeze(-1))
            )
        ).squeeze(-1)
        mask = torch.arange(seq.size(1), device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)

        gate = torch.sigmoid(self.knowledge_gate(torch.cat([context, local_summary, global_summary], dim=-1)))
        knowledge = gate * self.local_proj(local_summary) + (1.0 - gate) * self.global_proj(global_summary)
        return context, knowledge + summary


class KerPrintModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        group_indices: list[list[int]] | None = None,
        group_names: list[str] | None = None,
        global_adj: torch.Tensor | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_groups = len(group_names or group_indices or [])
        if num_groups <= 0 and global_adj is not None:
            num_groups = int(global_adj.size(0))
        if num_groups <= 0:
            num_groups = input_dim
        self.backbone = KerPrintBackbone(
            num_groups=num_groups,
            hidden_dim=hidden_dim,
            global_adj=global_adj,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        group_values: torch.Tensor | None = None,
        group_mask: torch.Tensor | None = None,
        visit_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context, knowledge = self.backbone(
            lengths,
            group_values=group_values,
            group_mask=group_mask,
            visit_time=visit_time,
        )
        return self.head(torch.cat([context, knowledge], dim=-1))


class KerPrintTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.core = KerPrintModel(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        group_values: torch.Tensor | None = None,
        group_mask: torch.Tensor | None = None,
        visit_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for t in range(x.size(1)):
            cur_lengths = lengths.clamp(max=t + 1).clamp(min=1)
            outputs.append(
                self.core(
                    x[:, : t + 1, :],
                    cur_lengths,
                    static,
                    group_values=None if group_values is None else group_values[:, : t + 1, :],
                    group_mask=None if group_mask is None else group_mask[:, : t + 1, :],
                    visit_time=None if visit_time is None else visit_time[:, : t + 1],
                )
            )
        return torch.stack(outputs, dim=1)
