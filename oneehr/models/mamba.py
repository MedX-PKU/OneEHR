"""EHR-Mamba model: state-space model for EHR sequences.

Ported from PyHealth's ehrmamba.py (RMSNorm + MambaBlock).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from oneehr.models.recurrent import last_by_lengths


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MambaBlock(nn.Module):
    """Selective state-space block (simplified Mamba).

    Implements the core Mamba mechanism: input projection → conv1d → SSM → output gate.
    """

    def __init__(
        self,
        d_model: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(d_inner, state_size * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # SSM parameters
        A = torch.arange(1, state_size + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        d_inner = xz.shape[-1] // 2
        x_inner, z = xz.split(d_inner, dim=-1)

        # Conv1d
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, : x_inner.shape[1]]  # trim to T
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters from input
        x_ssm = self.x_proj(x_conv)
        dt = F.softplus(self.dt_proj(x_ssm[:, :, -1:]))  # (B, T, d_inner)
        B_param = x_ssm[:, :, : self.state_size]  # (B, T, state_size)
        C_param = x_ssm[:, :, self.state_size : self.state_size * 2]  # (B, T, state_size)

        # Discretize and scan
        A = -torch.exp(self.A_log)  # (d_inner, state_size)
        batch, seq_len = x.shape[0], x.shape[1]

        y = torch.zeros(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(batch, d_inner, self.state_size, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            dt_t = dt[:, t]  # (B, d_inner)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, state_size)
            dB = dt_t.unsqueeze(-1) * B_param[:, t].unsqueeze(1)  # (B, d_inner, state_size)
            h = dA * h + dB * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            y[:, t] = y_t + self.D * x_conv[:, t]

        # Output gate
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class MambaEncoder(nn.Module):
    """Stack of MambaBlocks with input projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim, state_size=state_size, conv_kernel=conv_kernel) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.drop(self.norm(h))


class EHRMambaModel(nn.Module):
    """Patient-level Mamba: encoder → last valid step → head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 2,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MambaEncoder(input_dim, hidden_dim, num_layers, state_size, conv_kernel, dropout)
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


class EHRMambaTimeModel(nn.Module):
    """Time-level Mamba: encoder → head per-timestep."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 2,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MambaEncoder(input_dim, hidden_dim, num_layers, state_size, conv_kernel, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)
