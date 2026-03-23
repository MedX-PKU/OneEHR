"""GRASP model for patient-level and time-level prediction.

Paper: Liantao Ma et al. GRASP: Generic framework for health status
representation learning based on incorporating knowledge from similar
patients. AAAI 2021.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from oneehr.models.recurrent import last_by_lengths


def _generate_mask(lengths: torch.Tensor) -> torch.Tensor:
    max_len = int(lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _random_init(data: torch.Tensor, k: int) -> torch.Tensor:
    n = data.size(0)
    k = min(k, n)
    idx = torch.tensor(random.sample(range(n), k), dtype=torch.long)
    return data[idx]


def _cluster(data: torch.Tensor, k: int, max_iter: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple K-means clustering."""
    k = min(k, data.size(0))
    centers = _random_init(data, k)
    for _ in range(max_iter):
        dists = torch.cdist(data, centers)
        codes = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for j in range(k):
            mask = codes == j
            if mask.any():
                new_centers[j] = data[mask].mean(dim=0)
            else:
                new_centers[j] = centers[j]
        if torch.equal(centers, new_centers):
            break
        centers = new_centers
    return centers, codes


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        std = 1.0 / math.sqrt(out_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(adj.float(), torch.mm(x.float(), self.weight.float())) + self.bias


class GRASPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        cluster_num: int = 12,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.static_dim = static_dim

        # GRU backbone
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.weight1 = nn.Linear(hidden_dim, 1)
        self.weight2 = nn.Linear(hidden_dim, 1)
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        hidden_t = last_by_lengths(out, lengths)  # (B, H)

        # Clustering
        k = min(self.cluster_num, hidden_t.size(0))
        centers, _ = _cluster(hidden_t.detach(), k)
        centers = centers.to(hidden_t.device)

        adj = torch.eye(k, device=hidden_t.device)

        e = torch.relu(torch.mm(hidden_t, centers.t()))
        scores = torch.softmax(e, dim=-1)

        h_prime = torch.relu(self.gcn1(adj, centers))
        h_prime = torch.relu(self.gcn2(adj, h_prime))

        clu_app = torch.mm(scores, h_prime)

        w1 = torch.sigmoid(self.weight1(clu_app))
        w2 = torch.sigmoid(self.weight2(hidden_t))
        w1 = w1 / (w1 + w2)

        return w1 * clu_app + (1 - w1) * hidden_t


class GRASPModel(nn.Module):
    """Patient-level GRASP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        cluster_num: int = 12,
        out_dim: int = 1,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.layer = GRASPLayer(input_dim, hidden_dim, cluster_num, static_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.layer(x, lengths, static)
        return self.head(h)


class GRASPTimeModel(nn.Module):
    """Time-level GRASP: iterates with cumulative input."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        cluster_num: int = 12,
        out_dim: int = 1,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.layer = GRASPLayer(input_dim, hidden_dim, cluster_num, static_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        outputs = []
        for t in range(T):
            cur_x = x[:, : t + 1, :]
            cur_len = lengths.clamp(max=t + 1).clamp(min=1)
            h = self.layer(cur_x, cur_len, static)
            outputs.append(self.head(h))
        return torch.stack(outputs, dim=1)
