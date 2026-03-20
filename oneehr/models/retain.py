"""RETAIN model for patient-level and time-level prediction.

Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
Healthcare using Reverse Time Attention Mechanism. NIPS 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from oneehr.models.recurrent import last_by_lengths


class RETAINLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.alpha_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.alpha_fc = nn.Linear(hidden_dim, 1)
        self.beta_fc = nn.Linear(hidden_dim, input_dim)

    @staticmethod
    def _reverse(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        rev = x.clone()
        for i, l in enumerate(lengths.long().tolist()):
            if l > 0:
                rev[i, :l] = x[i, :l].flip(0)
        return rev

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Returns context vector of shape (B, input_dim)."""
        x = self.dropout(x)

        # Trim to max actual length to avoid dimension mismatch after pad_packed_sequence
        max_len = int(lengths.max().item())
        x = x[:, :max_len, :]

        rx = self._reverse(x, lengths)

        packed = rnn_utils.pack_padded_sequence(rx, lengths.cpu(), batch_first=True, enforce_sorted=False)

        g, _ = self.alpha_gru(packed)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        alpha = torch.softmax(self.alpha_fc(g), dim=1)  # (B, T, 1)

        h, _ = self.beta_gru(packed)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        beta = torch.tanh(self.beta_fc(h))  # (B, T, input_dim)

        c = (alpha * beta * x).sum(dim=1)  # (B, input_dim)
        return c


class RETAINModel(nn.Module):
    """Patient-level RETAIN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layer = RETAINLayer(input_dim, hidden_dim, dropout)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.layer(x, lengths)
        return self.head(torch.relu(self.proj(c)))


class RETAINTimeModel(nn.Module):
    """Time-level RETAIN: iterates with cumulative input."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layer = RETAINLayer(input_dim, hidden_dim, dropout)
        self.proj = nn.Linear(input_dim, hidden_dim)
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
            cur_len = lengths.clamp(max=t + 1)
            cur_len = cur_len.clamp(min=1)
            c = self.layer(cur_x, cur_len)
            outputs.append(self.head(torch.relu(self.proj(c))))
        return torch.stack(outputs, dim=1)
