from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class SequenceBatch:
    x: torch.Tensor  # (B, T, D)
    lengths: torch.Tensor  # (B,)
    y: torch.Tensor  # (B,) or (B, T)
    mask: torch.Tensor | None  # same shape as y


def build_patient_sequences(binned: pd.DataFrame, feature_columns: list[str]):
    """Build variable-length sequences per patient from binned table."""

    required = {"patient_id", "bin_time"}
    missing = [c for c in required if c not in binned.columns]
    if missing:
        raise ValueError(f"binned missing columns: {missing}")

    df = binned[["patient_id", "bin_time", *feature_columns]].copy()
    df = df.sort_values(["patient_id", "bin_time"], kind="stable")
    groups = list(df.groupby("patient_id", sort=False))
    patient_ids = [str(pid) for pid, _ in groups]
    seqs = [g[feature_columns].to_numpy(dtype=np.float32) for _, g in groups]
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    return patient_ids, seqs, lengths


def pad_sequences(seqs: list[np.ndarray], lengths: np.ndarray) -> torch.Tensor:
    max_len = int(lengths.max()) if len(lengths) else 0
    if max_len == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)
    feat_dim = int(seqs[0].shape[1])
    out = np.zeros((len(seqs), max_len, feat_dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0], :] = s
    return torch.from_numpy(out)

