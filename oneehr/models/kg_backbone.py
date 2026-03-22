"""Shared graph backbones for KG-enhanced EHR models."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.graph import GraphConvolution, normalize_adjacency


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = torch.where(mask.any(dim=dim, keepdim=True), scores, torch.zeros_like(scores))
    weights = torch.softmax(scores, dim=dim)
    return torch.where(mask, weights, torch.zeros_like(weights))


class ConceptGraphEncoder(nn.Module):
    def __init__(self, num_groups: int, hidden_dim: int, global_adj: torch.Tensor | None = None):
        super().__init__()
        adj = torch.eye(num_groups, dtype=torch.float32) if global_adj is None else global_adj.detach().to(torch.float32)
        self.register_buffer("global_adj", normalize_adjacency(adj))
        self.node_embed = nn.Parameter(torch.randn(num_groups, hidden_dim) * 0.02)
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)

    def forward(self) -> torch.Tensor:
        h = torch.relu(self.gcn1(self.global_adj, self.node_embed))
        return torch.relu(self.gcn2(self.global_adj, h))


class KGVisitEncoder(nn.Module):
    def __init__(self, num_groups: int, hidden_dim: int, global_adj: torch.Tensor | None = None, dropout: float = 0.1):
        super().__init__()
        self.graph = ConceptGraphEncoder(num_groups, hidden_dim, global_adj=global_adj)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        group_values: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = self.graph()
        tokens = self.value_proj(group_values.unsqueeze(-1)) + nodes.view(1, 1, nodes.size(0), nodes.size(1))
        mask = group_mask > 0
        weights = masked_softmax(self.score_proj(torch.tanh(tokens)).squeeze(-1), mask, dim=-1)
        visits = (weights.unsqueeze(-1) * self.dropout(tokens)).sum(dim=2)
        patient_mask = mask.any(dim=1)
        return visits, nodes, patient_mask


class PatientGraphSummary(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.local_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, nodes: torch.Tensor, patient_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_summary = nodes.mean(dim=0, keepdim=True).expand(patient_mask.size(0), -1)
        masked_nodes = patient_mask.to(dtype=nodes.dtype).unsqueeze(-1) * nodes.unsqueeze(0)
        denom = patient_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=nodes.dtype)
        local_summary = masked_nodes.sum(dim=1) / denom
        fused = torch.sigmoid(self.gate(torch.cat([local_summary, global_summary], dim=-1)))
        summary = fused * self.local_proj(local_summary) + (1.0 - fused) * self.global_proj(global_summary)
        return summary, local_summary, global_summary
