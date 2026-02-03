from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _last_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    idx = (lengths - 1).clamp_min(0)
    return x[torch.arange(x.shape[0], device=x.device), idx]


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        out = self.encoder(x, lengths)
        last = _last_by_lengths(out, lengths)
        return self.head(last)


class LSTMTimeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        out = self.encoder(x, lengths)
        return self.head(out)


@dataclass(frozen=True)
class LSTMArtifacts:
    feature_columns: list[str]
    state_dict: dict

