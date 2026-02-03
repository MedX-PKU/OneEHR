from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _last_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    idx = (lengths - 1).clamp_min(0)
    return x[torch.arange(x.shape[0], device=x.device), idx]


class DrAgentModel(nn.Module):
    """A simplified 'Dr. Agent' sequence model with optional static features.

    This is a structure-clean rewrite aligned with OneEHR's trainer contract:
    - patient mode: forward(x, lengths, static=None) -> (B, 1)
    - time mode: forward(x, lengths, static=None) -> (B, T, 1)

    Notes:
    - PyEHR's Agent contains a more complex RL-inspired module; this MVP keeps the
      same "dynamic + static" interface while we port the full architecture.
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_proj = nn.Identity() if self.static_dim == 0 else nn.Linear(static_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * (2 if self.static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        last = _last_by_lengths(out, lengths)
        if static is None or self.static_dim == 0:
            return self.head(last)
        s = self.static_proj(static)
        return self.head(torch.cat([last, s], dim=-1))


class DrAgentTimeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_proj = nn.Identity() if self.static_dim == 0 else nn.Linear(static_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim * (2 if self.static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if static is None or self.static_dim == 0:
            return self.head(out)
        s = self.static_proj(static).unsqueeze(1).expand(out.shape[0], out.shape[1], -1)
        return self.head(torch.cat([out, s], dim=-1))


@dataclass(frozen=True)
class DrAgentArtifacts:
    feature_columns: list[str]
    state_dict: dict

