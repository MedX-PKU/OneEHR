"""ProtoEHR-style prototype augmented KG models."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.kg_backbone import KGVisitEncoder, PatientGraphSummary
from oneehr.models.recurrent import last_by_lengths
from oneehr.models.time import ContinuousTimeEncoding


class PrototypeMemory(nn.Module):
    def __init__(self, num_prototypes: int, hidden_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim) * 0.02)
        self.query = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.query(x), self.prototypes.transpose(0, 1))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, self.prototypes)


class ProtoEHRBackbone(nn.Module):
    def __init__(
        self,
        num_groups: int,
        hidden_dim: int = 128,
        num_prototypes: int = 8,
        global_adj: torch.Tensor | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visit_encoder = KGVisitEncoder(num_groups, hidden_dim, global_adj=global_adj, dropout=dropout)
        self.graph_summary = PatientGraphSummary(hidden_dim)
        self.time_enc = ContinuousTimeEncoding(hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.concept_memory = PrototypeMemory(num_prototypes, hidden_dim)
        self.visit_memory = PrototypeMemory(num_prototypes, hidden_dim)
        self.patient_memory = PrototypeMemory(num_prototypes, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        lengths: torch.Tensor,
        *,
        group_values: torch.Tensor | None,
        group_mask: torch.Tensor | None,
        visit_time: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if group_values is None or group_mask is None:
            raise ValueError("ProtoEHR requires group_values and group_mask in forward()")
        visits, nodes, patient_mask = self.visit_encoder(group_values, group_mask)
        summary, local_summary, _ = self.graph_summary(nodes, patient_mask)
        if visit_time is not None:
            visits = visits + self.time_enc(visit_time)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.dropout(visits), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.temporal_gru(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        patient_repr = last_by_lengths(seq, lengths)
        concept_proto = self.concept_memory(local_summary)
        visit_proto = self.visit_memory(seq.mean(dim=1))
        patient_proto = self.patient_memory(patient_repr + summary)
        return patient_repr, concept_proto, visit_proto, patient_proto


class ProtoEHRModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_prototypes: int = 8,
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
        self.backbone = ProtoEHRBackbone(
            num_groups=num_groups,
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            global_adj=global_adj,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim * 4, out_dim)

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
        patient_repr, concept_proto, visit_proto, patient_proto = self.backbone(
            lengths,
            group_values=group_values,
            group_mask=group_mask,
            visit_time=visit_time,
        )
        fused = torch.cat([patient_repr, concept_proto, visit_proto, patient_proto], dim=-1)
        return self.head(fused)


class ProtoEHRTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.core = ProtoEHRModel(**kwargs)

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
