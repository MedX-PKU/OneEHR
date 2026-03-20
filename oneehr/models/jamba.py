"""Jamba-EHR model: interleaved Transformer + Mamba layers.

Ported from PyHealth's jamba_ehr.py (build_layer_schedule).
"""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.mamba import MambaBlock, RMSNorm
from oneehr.models.recurrent import last_by_lengths


def build_layer_schedule(num_transformer_layers: int, num_mamba_layers: int) -> list[str]:
    """Build an interleaved schedule of 'transformer' and 'mamba' layer types.

    Distributes transformer layers evenly among mamba layers.
    """
    total = num_transformer_layers + num_mamba_layers
    if total == 0:
        return []
    schedule: list[str] = []
    t_placed = 0
    m_placed = 0
    for i in range(total):
        # Place transformer layer at evenly-spaced positions
        t_target = (i + 1) * num_transformer_layers / total
        if t_placed < t_target and t_placed < num_transformer_layers:
            schedule.append("transformer")
            t_placed += 1
        else:
            schedule.append("mamba")
            m_placed += 1
    return schedule


class JambaEncoder(nn.Module):
    """Interleaved Transformer + Mamba encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_transformer_layers: int = 2,
        num_mamba_layers: int = 6,
        heads: int = 4,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        schedule = build_layer_schedule(num_transformer_layers, num_mamba_layers)
        self.layers = nn.ModuleList()
        for kind in schedule:
            if kind == "transformer":
                self.layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=heads,
                        dim_feedforward=hidden_dim * 4,
                        dropout=dropout,
                        batch_first=True,
                    )
                )
            else:
                self.layers.append(
                    MambaBlock(hidden_dim, state_size=state_size, conv_kernel=conv_kernel)
                )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class JambaModel(nn.Module):
    """Patient-level Jamba: encoder → last valid step → head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_transformer_layers: int = 2,
        num_mamba_layers: int = 6,
        heads: int = 4,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = JambaEncoder(
            input_dim, hidden_dim, num_transformer_layers, num_mamba_layers,
            heads, state_size, conv_kernel, dropout,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)
        last = last_by_lengths(h, lengths)
        return self.head(last)


class JambaTimeModel(nn.Module):
    """Time-level Jamba: encoder → head per-timestep."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_transformer_layers: int = 2,
        num_mamba_layers: int = 6,
        heads: int = 4,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = JambaEncoder(
            input_dim, hidden_dim, num_transformer_layers, num_mamba_layers,
            heads, state_size, conv_kernel, dropout,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)
