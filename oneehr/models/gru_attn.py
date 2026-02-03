from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class AttnGRUEncoder(nn.Module):
    """GRU encoder with simple temporal attention.

    Attention here is over time steps (T). It returns:
    - pooled embedding: (B, H)
    - attention weights: (B, T)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,H)

        B, T, _H = out.shape
        device = out.device
        t_idx = torch.arange(T, device=device)[None, :].expand(B, T)
        valid = t_idx < lengths[:, None]  # (B,T)

        scores = self.attn(out).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(~valid, float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)  # (B,T)

        pooled = (out * attn_weights[:, :, None]).sum(dim=1)  # (B,H)
        return pooled, attn_weights


class AttnGRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = AttnGRUEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb, _attn = self.encoder(x, lengths)
        return self.head(emb)

    def forward_with_attention(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb, attn = self.encoder(x, lengths)
        logits = self.head(emb)
        return logits, attn


@dataclass(frozen=True)
class AttnGRUArtifacts:
    feature_columns: list[str]
    state_dict: dict

