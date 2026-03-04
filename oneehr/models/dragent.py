from __future__ import annotations

import torch
from torch import nn

from oneehr.models.utils import last_by_lengths


class _DrAgentEncoder(nn.Module):
    """Shared encoder for DrAgent patient and time modes."""

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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out


class DrAgentModel(nn.Module):
    """Dr. Agent model with optional static features — patient-level (N-1)."""

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
        self.enc = _DrAgentEncoder(input_dim, static_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Linear(hidden_dim * (2 if static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        out = self.enc(x, lengths)
        last = last_by_lengths(out, lengths)
        if static is None or self.static_dim == 0:
            return self.head(last)
        s = self.enc.static_proj(static)
        return self.head(torch.cat([last, s], dim=-1))


class DrAgentTimeModel(nn.Module):
    """Dr. Agent model with optional static features — time-level (N-N)."""

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
        self.enc = _DrAgentEncoder(input_dim, static_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Linear(hidden_dim * (2 if static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        out = self.enc(x, lengths)
        if static is None or self.static_dim == 0:
            return self.head(out)
        s = self.enc.static_proj(static).unsqueeze(1).expand(out.shape[0], out.shape[1], -1)
        return self.head(torch.cat([out, s], dim=-1))
