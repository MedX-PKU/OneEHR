"""Unified GRU/LSTM recurrent models with a ``cell`` parameter."""

from __future__ import annotations

import torch
from torch import nn


def _make_rnn(cell: str, **kwargs) -> nn.GRU | nn.LSTM | nn.RNN:
    if cell == "gru":
        return nn.GRU(**kwargs)
    if cell == "lstm":
        return nn.LSTM(**kwargs)
    if cell == "rnn":
        return nn.RNN(**kwargs)
    raise ValueError(f"Unsupported cell={cell!r}")


def last_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Select the last valid timestep for each sequence.

    Args:
        x: Tensor of shape (B, T, D)
        lengths: Tensor of shape (B,) containing valid lengths (>=0)
    """
    idx = (lengths - 1).clamp_min(0)
    return x[torch.arange(x.shape[0], device=x.device), idx]


class RecurrentModel(nn.Module):
    """Patient-level recurrent model (GRU or LSTM)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        cell: str = "gru",
    ):
        super().__init__()
        self.cell = cell
        self.rnn = _make_rnn(
            cell,
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        last = last_by_lengths(out, lengths)
        return self.head(last)


class RecurrentTimeModel(nn.Module):
    """Time-level recurrent model (GRU or LSTM)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        cell: str = "gru",
    ):
        super().__init__()
        self.cell = cell
        self.rnn = _make_rnn(
            cell,
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.head(out)


# Backward-compatible aliases
GRUModel = RecurrentModel
GRUTimeModel = RecurrentTimeModel
LSTMModel = RecurrentModel
LSTMTimeModel = RecurrentTimeModel
