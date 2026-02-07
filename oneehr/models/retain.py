from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class RETAINLayer(nn.Module):
    """RETAIN layer."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_layer = nn.Dropout(p=float(dropout))

        self.alpha_gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.beta_gru = nn.GRU(input_dim, input_dim, batch_first=True)

        self.alpha_li = nn.Linear(input_dim, 1)
        self.beta_li = nn.Linear(input_dim, input_dim)

    @staticmethod
    def _reverse_x(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        reversed_x = x.new_zeros(x.size())
        for i, length in enumerate(lengths.tolist()):
            reversed_x[i, :length] = x[i, :length].flip(dims=[0])
        return reversed_x

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout_layer(x)
        rx = self._reverse_x(x, lengths)

        packed = nn.utils.rnn.pack_padded_sequence(
            rx, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        g, _ = self.alpha_gru(packed)
        g, _ = nn.utils.rnn.pad_packed_sequence(g, batch_first=True)
        alpha = torch.softmax(self.alpha_li(g), dim=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            rx, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        beta = torch.tanh(self.beta_li(h))

        c = alpha * beta * x
        return torch.sum(c, dim=1)


class RETAINModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.layer = RETAINLayer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        c = self.layer(x, lengths)
        return self.head(c)


class RETAINTimeModel(nn.Module):
    """Simplified N-N RETAIN model.

    We reuse the RETAIN layer per-prefix by computing a cumulative mask. This is
    O(T^2) and meant as a correct baseline; we can optimize later.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.layer = RETAINLayer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        bsz, t, _ = x.shape
        device = x.device
        out = x.new_zeros((bsz, t, 1))
        for cur_t in range(t):
            cur_len = torch.clamp(lengths, max=cur_t + 1)
            c = self.layer(x[:, : cur_t + 1], cur_len)
            out[:, cur_t, :] = self.head(c)
        # zero-out padded steps (trainer uses mask anyway)
        pad_mask = torch.arange(t, device=device)[None, :] >= lengths[:, None]
        out[pad_mask] = 0.0
        return out


@dataclass(frozen=True)
class RETAINArtifacts:
    feature_columns: list[str]
    state_dict: dict
