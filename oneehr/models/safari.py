"""SAFARI model family.

This implementation keeps the reference model's MCGRU + attention flavour
while removing dataset-specific shape assumptions.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from oneehr.models.graph import normalize_adjacency
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


class FeatureGraphRefiner(nn.Module):
    """Cluster-aware feature graph refinement adapted from SAFARI."""

    def __init__(self, hidden_dim: int, n_clu: int = 8):
        super().__init__()
        self.n_clu = n_clu
        self.gcn_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_w2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def _build_adjacency(self, tokens: torch.Tensor) -> torch.Tensor:
        node_repr = tokens.mean(dim=0)
        num_nodes = node_repr.size(0)
        if num_nodes <= 1:
            return torch.eye(num_nodes, device=tokens.device, dtype=tokens.dtype)

        k = min(self.n_clu, num_nodes)
        labels = KMeans(n_clusters=k, init="random", n_init=2, random_state=42).fit_predict(
            node_repr.detach().cpu().numpy()
        )
        adj = torch.zeros(num_nodes, num_nodes, device=tokens.device, dtype=tokens.dtype)
        norm_repr = F.normalize(node_repr, dim=-1)
        sim = torch.matmul(norm_repr, norm_repr.transpose(0, 1)).clamp_min(0.0)

        for clu in range(k):
            idx = torch.as_tensor([i for i, label in enumerate(labels) if label == clu], device=tokens.device)
            if idx.numel() == 0:
                continue
            adj[idx.unsqueeze(1), idx.unsqueeze(0)] = sim[idx.unsqueeze(1), idx.unsqueeze(0)]

        return normalize_adjacency(adj)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        adj = self._build_adjacency(tokens)
        h = self.relu(self.gcn_w1(torch.matmul(adj.unsqueeze(0), tokens)))
        h = self.relu(self.gcn_w2(torch.matmul(adj.unsqueeze(0), h)))
        return h


class SafariBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dim_list: list[int] | None = None,
        n_clu: int = 8,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.dim_list = list(dim_list or [1] * input_dim)
        self.encoder = SafariMCGRU(self.dim_list, hidden_dim)
        self.graph_refiner = FeatureGraphRefiner(hidden_dim, n_clu=n_clu)
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
        graph_tokens = self.graph_refiner(self.dropout(tokens))
        context = self.attention(graph_tokens)
        return self.relu(self.output_proj(self.dropout(context)))


class SafariModel(nn.Module):
    """Patient-level SAFARI."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 1,
        dim_list: list[int] | None = None,
        n_clu: int = 8,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.backbone = SafariBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dim_list=dim_list,
            n_clu=n_clu,
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
