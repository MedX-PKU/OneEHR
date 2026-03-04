from __future__ import annotations

import numpy as np
import torch
from torch import nn

from oneehr.models.utils import last_by_lengths


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / np.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-std, std)
            if self.bias is not None:
                self.bias.uniform_(-std, std)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        out = adj @ y
        if self.bias is not None:
            out = out + self.bias
        return out


def _kmeans_simple(x: torch.Tensor, k: int, iters: int = 30) -> torch.Tensor:
    # x: (N, D)
    n = x.shape[0]
    if k > n:
        k = n
    # init centers
    idx = torch.randperm(n, device=x.device)[:k]
    centers = x[idx].clone()
    for _ in range(iters):
        # assign
        dist = torch.cdist(x, centers)
        codes = torch.argmin(dist, dim=1)
        # update
        new_centers = []
        for j in range(k):
            m = codes == j
            if m.any():
                new_centers.append(x[m].mean(dim=0))
            else:
                new_centers.append(centers[j])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


class GRASPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, cluster_num: int = 12,
                 static_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cluster_num = int(cluster_num)
        self.static_dim = int(static_dim)

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.weight1 = nn.Linear(hidden_dim, 1)
        self.weight2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.relu = nn.ReLU()
        self.static_proj = nn.Identity() if self.static_dim == 0 else nn.Linear(static_dim, hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)
        hidden = h_n[-1]  # (B, H)

        centers = _kmeans_simple(hidden, self.cluster_num)
        # fully-connected cluster graph (simple default)
        k = centers.shape[0]
        adj = torch.eye(k, device=x.device, dtype=centers.dtype)

        e = self.relu(hidden @ centers.transpose(0, 1))  # (B, K)
        scores = torch.softmax(e, dim=-1)

        h_prime = self.relu(self.gcn1(adj, centers))
        h_prime = self.relu(self.gcn2(adj, h_prime))

        clu = scores @ h_prime  # (B, H)
        w1 = torch.sigmoid(self.weight1(clu))
        w2 = torch.sigmoid(self.weight2(hidden))
        w1 = w1 / (w1 + w2 + 1e-8)
        out = w1 * clu + (1.0 - w1) * hidden
        return self.dropout(out)


class GRASPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, cluster_num: int = 12,
                 static_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.static_dim = int(static_dim)
        self.encoder = GRASPEncoder(input_dim, hidden_dim=hidden_dim, cluster_num=cluster_num,
                                    static_dim=static_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim * (2 if static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x, lengths)
        if static is not None and self.static_dim > 0:
            s = self.encoder.static_proj(static)
            z = torch.cat([z, s], dim=-1)
        return self.head(z)


class GRASPTimeModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, cluster_num: int = 12,
                 static_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.static_dim = int(static_dim)
        self.encoder = GRASPEncoder(input_dim, hidden_dim=hidden_dim, cluster_num=cluster_num,
                                    static_dim=static_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim * (2 if static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        bsz, t, _ = x.shape
        s_proj = None
        if static is not None and self.static_dim > 0:
            s_proj = self.encoder.static_proj(static)
        out = x.new_zeros((bsz, t, 1))
        for cur_t in range(t):
            cur_len = torch.clamp(lengths, max=cur_t + 1)
            z = self.encoder(x[:, : cur_t + 1], cur_len)
            if s_proj is not None:
                z = torch.cat([z, s_proj], dim=-1)
            out[:, cur_t, :] = self.head(z)
        pad_mask = torch.arange(t, device=x.device)[None, :] >= lengths[:, None]
        out[pad_mask] = 0.0
        return out
