from __future__ import annotations

import torch


def last_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Select the last valid timestep for each sequence.

    Args:
        x: Tensor of shape (B, T, D)
        lengths: Tensor of shape (B,) containing valid lengths (>=0)
    """

    idx = (lengths - 1).clamp_min(0)
    return x[torch.arange(x.shape[0], device=x.device), idx]

