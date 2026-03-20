"""ConCare model for patient-level and time-level prediction.

Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding
via capturing the healthcare context. AAAI 2020.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from oneehr.models.recurrent import last_by_lengths


def _generate_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create (B, T) boolean mask from lengths."""
    max_len = int(lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class FinalAttentionQKV(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.5):
        super().__init__()
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x[:, -1, :]).unsqueeze(1)  # (B, 1, D)
        k = self.W_k(x)  # (B, N, D)
        v = self.W_v(x)  # (B, N, D)
        e = self.W_out(torch.tanh(q + k)).squeeze(-1)  # (B, N)
        a = self.dropout(torch.softmax(e, dim=-1))
        return torch.bmm(a.unsqueeze(1), v).squeeze(1)  # (B, D)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, _ = q.size()
        q = self.q_proj(q).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class ConCareLayer(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim

        # Channel-wise GRUs
        self.grus = nn.ModuleList([
            nn.GRU(1, hidden_dim, batch_first=True) for _ in range(lab_dim)
        ])

        if static_dim > 0:
            self.static_proj = nn.Linear(static_dim, hidden_dim)

        self.mha = MultiHeadAttention(num_heads, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.final_attn = FinalAttentionQKV(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.size()
        # Channel-wise encoding
        embeddings = []
        for i in range(D):
            feat_i = x[:, :, i].unsqueeze(-1)  # (B, T, 1)
            out_i, _ = self.grus[i](feat_i)  # (B, T, H)
            # Last valid timestep per channel
            last_i = last_by_lengths(out_i, mask.sum(dim=1))  # (B, H)
            embeddings.append(last_i)

        emb = torch.stack(embeddings, dim=1)  # (B, D, H)

        if static is not None and self.static_dim > 0:
            s = torch.tanh(self.static_proj(static)).unsqueeze(1)  # (B, 1, H)
            emb = torch.cat([emb, s], dim=1)  # (B, D+1, H)

        emb = self.dropout(emb)
        # Self-attention
        h = self.norm1(emb + self.mha(emb, emb, emb))
        h = self.norm2(h + self.ffn(h))
        # Final attention pooling
        return self.final_attn(h)  # (B, H)


class ConCareModel(nn.Module):
    """Patient-level ConCare."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        out_dim: int = 1,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.layer = ConCareLayer(input_dim, hidden_dim, num_heads, static_dim, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = _generate_mask(lengths)
        c = self.layer(x, mask, static)
        return self.head(c)


class ConCareTimeModel(nn.Module):
    """Time-level ConCare: iterates with cumulative input."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        out_dim: int = 1,
        static_dim: int = 0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.layer = ConCareLayer(input_dim, hidden_dim, num_heads, static_dim, dropout)
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
            cur_x = x[:, :t + 1, :]
            cur_len = lengths.clamp(max=t + 1).clamp(min=1)
            cur_mask = _generate_mask(cur_len)
            c = self.layer(cur_x, cur_mask, static)
            outputs.append(self.head(c))
        return torch.stack(outputs, dim=1)
