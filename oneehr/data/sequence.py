from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import torch


def _truncate_seqs(
    seqs: list[np.ndarray], lengths: np.ndarray, max_len: int | None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Truncate sequences to at most *max_len* time steps (keep most recent)."""
    if max_len is None or max_len <= 0:
        return seqs, lengths
    new_seqs = []
    new_lengths = np.empty(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        if s.shape[0] > max_len:
            new_seqs.append(s[-max_len:])
            new_lengths[i] = max_len
        else:
            new_seqs.append(s)
            new_lengths[i] = s.shape[0]
    return new_seqs, new_lengths


def build_patient_sequences(
    binned: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_seq_length: int | None = None,
):
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
    seqs, lengths = _truncate_seqs(seqs, lengths, max_seq_length)
    return patient_ids, seqs, lengths


def align_static_features(
    patient_ids: list[str],
    static_df: pd.DataFrame | None,
    *,
    expected_feature_columns: list[str] | None = None,
) -> np.ndarray | None:
    """Align static (patient-level) features to a sequence batch.

    Returns a float32 numpy array of shape (N, S) or None if no static features.
    """

    if static_df is None or static_df.empty:
        return None
    df = static_df.copy()
    df.index = df.index.astype(str)
    df = df.reindex(patient_ids)
    if expected_feature_columns is not None:
        df = df.reindex(columns=list(expected_feature_columns))
    if df.isna().any().any():
        df = df.fillna(0.0)
    return df.to_numpy(dtype=np.float32)


def build_time_sequences(
    binned: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_time_col: str = "bin_time",
    max_seq_length: int | None = None,
):
    """Build variable-length sequences per patient with N-N labels.

    Returns:
    - patient_ids: list[str]
    - time_seqs: list[np.ndarray] of shape (T_i,) containing bin_time values
    - seqs: list[np.ndarray] of shape (T_i, D)
    - y_seqs: list[np.ndarray] of shape (T_i,)
    - mask_seqs: list[np.ndarray] of shape (T_i,) where 1.0 means valid label
    - lengths: np.ndarray (N,)

    Notes:
    - `binned` is the binned feature table (long) with patient_id/bin_time and feature columns.
    - `labels` should contain columns: patient_id, label, and a time column (default `bin_time`).
      It may also contain `mask` (bool) to indicate which labels are valid.
    """

    required_b = {"patient_id", "bin_time", *feature_columns}
    missing_b = [c for c in required_b if c not in binned.columns]
    if missing_b:
        raise ValueError(f"binned missing columns: {missing_b}")

    if "patient_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError("labels must contain patient_id and label")
    if label_time_col not in labels.columns:
        raise ValueError(f"labels missing time column: {label_time_col!r}")

    feat = binned[["patient_id", "bin_time", *feature_columns]].copy()
    feat = feat.sort_values(["patient_id", "bin_time"], kind="stable")

    lab_cols = ["patient_id", label_time_col, "label"]
    if "mask" in labels.columns:
        lab_cols.append("mask")
    lab = labels[lab_cols].copy()
    lab = lab.rename(columns={label_time_col: "bin_time"})
    lab = lab.sort_values(["patient_id", "bin_time"], kind="stable")

    df = feat.merge(lab, on=["patient_id", "bin_time"], how="left")
    if "mask" in df.columns:
        valid = df["mask"].fillna(False).to_numpy(dtype=bool)
    else:
        valid = df["label"].notna().to_numpy(dtype=bool)

    df["_mask"] = valid
    df["_label"] = df["label"].fillna(0.0)

    groups = list(df.groupby("patient_id", sort=False))
    patient_ids = [str(pid) for pid, _ in groups]
    time_seqs = [g["bin_time"].to_numpy() for _, g in groups]
    seqs = [g[feature_columns].to_numpy(dtype=np.float32) for _, g in groups]
    y_seqs = [g["_label"].to_numpy(dtype=np.float32) for _, g in groups]
    mask_seqs = [g["_mask"].to_numpy(dtype=np.float32) for _, g in groups]
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)

    # Truncate to most recent max_seq_length bins.
    if max_seq_length is not None and max_seq_length > 0:
        for i in range(len(seqs)):
            if seqs[i].shape[0] > max_seq_length:
                seqs[i] = seqs[i][-max_seq_length:]
                y_seqs[i] = y_seqs[i][-max_seq_length:]
                mask_seqs[i] = mask_seqs[i][-max_seq_length:]
                time_seqs[i] = time_seqs[i][-max_seq_length:]
                lengths[i] = max_seq_length

    return patient_ids, time_seqs, seqs, y_seqs, mask_seqs, lengths


def pad_sequences(seqs: list[np.ndarray], lengths: np.ndarray):
    max_len = int(lengths.max()) if len(lengths) else 0
    if max_len == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)
    feat_dim = int(seqs[0].shape[1])
    out = np.zeros((len(seqs), max_len, feat_dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0], :] = s
    # Fill any NaNs in the padded tensor (some datasets may still contain NaNs
    # after preprocessing, and they can poison DL training/inference).
    if np.isnan(out).any():
        nan_count = int(np.isnan(out).sum())
        warnings.warn(
            f"pad_sequences: {nan_count} residual NaN values found after preprocessing. "
            "Consider adding an impute step to your pipeline config. "
            "Filling with 0 as safety net.",
            stacklevel=2,
        )
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure the returned tensor is writable/contiguous.
    return torch.from_numpy(out.copy())
