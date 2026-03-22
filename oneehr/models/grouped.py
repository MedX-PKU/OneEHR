"""Grouped visit encoders shared by visit-centric EHR models."""

from __future__ import annotations

import torch
from torch import nn


def resolve_group_indices(input_dim: int, group_indices: list[list[int]] | None) -> list[list[int]]:
    if group_indices:
        return [list(group) for group in group_indices if group]
    return [[idx] for idx in range(input_dim)]


class GroupedVisitEncoder(nn.Module):
    """Aggregate encoded feature groups into visit-level embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        group_indices: list[list[int]] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.group_indices = resolve_group_indices(input_dim, group_indices)
        self.group_proj = nn.Linear(1, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    @property
    def num_groups(self) -> int:
        return len(self.group_indices)

    def group_values(self, x: torch.Tensor) -> torch.Tensor:
        blocks = [x[:, :, idx].mean(dim=-1) for idx in self.group_indices]
        return torch.stack(blocks, dim=2)

    def group_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_proj(self.group_values(x).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.group_tokens(x)
        scores = self.score_proj(torch.tanh(tokens)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        visit = (weights.unsqueeze(-1) * self.dropout(tokens)).sum(dim=2)
        return visit, tokens, weights
