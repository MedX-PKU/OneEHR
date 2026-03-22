"""Shared sequence adapters for missingness, irregular time, and grouped features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from oneehr.data.sequence import build_patient_sequences
from oneehr.utils import parse_bin_size


@dataclass(frozen=True)
class FeatureGroup:
    """Logical feature group backed by one or more encoded columns."""

    name: str
    cols: list[str]
    indices: list[int]


def load_feature_schema(run_dir: Path) -> list[dict] | None:
    path = Path(run_dir) / "preprocess" / "feature_schema.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_obs_mask(run_dir: Path) -> pd.DataFrame | None:
    path = Path(run_dir) / "preprocess" / "obs_mask.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def normalize_feature_name(name: str) -> str:
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("cat__"):
        return name[5:]
    return name


def resolve_feature_groups(
    *,
    feat_cols: list[str],
    feature_schema: list[dict] | None,
) -> list[FeatureGroup]:
    feat_index = {col: i for i, col in enumerate(feat_cols)}
    groups: list[FeatureGroup] = []
    used: set[str] = set()

    for entry in feature_schema or []:
        cols = [str(col) for col in entry.get("cols", []) if str(col) in feat_index]
        if not cols:
            continue
        used.update(cols)
        groups.append(
            FeatureGroup(
                name=str(entry.get("name", normalize_feature_name(cols[0]))),
                cols=cols,
                indices=[feat_index[col] for col in cols],
            )
        )

    for col in feat_cols:
        if col in used:
            continue
        groups.append(
            FeatureGroup(
                name=normalize_feature_name(col),
                cols=[col],
                indices=[feat_index[col]],
            )
        )

    return groups


def _sorted_feature_frame(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in feat_cols if c in df.columns]
    return df[["patient_id", "bin_time", *cols]].copy().sort_values(
        ["patient_id", "bin_time"],
        kind="stable",
    )


def _infer_max_len(
    *,
    patient_ids: list[str],
    seq_map: dict[str, np.ndarray],
    max_len: int | None,
) -> int:
    if max_len is not None:
        return int(max_len)
    return max((seq_map.get(pid, np.zeros((0,), dtype=np.float32)).shape[0] for pid in patient_ids), default=0)


def build_missing_mask_tensor(
    *,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: list[str] | None = None,
    max_len: int | None = None,
) -> torch.Tensor:
    obs = _sorted_feature_frame(obs_mask, feat_cols)
    seq_pids, seqs, _ = build_patient_sequences(obs, feat_cols)
    seq_map = {
        str(pid): (1.0 - seq.astype(np.float32, copy=False))
        for pid, seq in zip(seq_pids, seqs, strict=True)
    }

    aligned_pids = [str(pid) for pid in (patient_ids or list(seq_pids))]
    out_max_len = _infer_max_len(patient_ids=aligned_pids, seq_map=seq_map, max_len=max_len)
    out = np.zeros((len(aligned_pids), out_max_len, len(feat_cols)), dtype=np.float32)

    for i, pid in enumerate(aligned_pids):
        seq = seq_map.get(pid)
        if seq is None or out_max_len <= 0:
            continue
        if seq.shape[0] > out_max_len:
            seq = seq[-out_max_len:]
        out[i, : seq.shape[0], :] = seq
    return torch.from_numpy(out)


def build_visit_time_map(
    *,
    binned: pd.DataFrame,
    patient_ids: set[str] | None = None,
    bin_size: str = "1d",
) -> dict[str, np.ndarray]:
    frame = binned[["patient_id", "bin_time"]].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if patient_ids is not None:
        frame = frame[frame["patient_id"].isin(set(str(pid) for pid in patient_ids))].copy()
    frame = frame.sort_values(["patient_id", "bin_time"], kind="stable")

    td_unit = pd.Timedelta(parse_bin_size(bin_size))
    out: dict[str, np.ndarray] = {}
    for pid, grp in frame.groupby("patient_id", sort=False):
        times = pd.to_datetime(grp["bin_time"])
        base = times.iloc[0]
        rel = ((times - base) / td_unit).astype(float).to_numpy(dtype=np.float32)
        out[str(pid)] = rel
    return out


def build_visit_time_tensor(
    visit_time_map: dict[str, np.ndarray],
    patient_ids: list[str],
    max_len: int,
) -> torch.Tensor:
    out = np.zeros((len(patient_ids), max_len), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        seq = visit_time_map.get(str(pid))
        if seq is None or max_len <= 0:
            continue
        if seq.shape[0] > max_len:
            seq = seq[-max_len:]
        out[i, : seq.shape[0]] = seq
    return torch.from_numpy(out)


def build_time_delta_map(
    *,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    bin_size: str = "1d",
    patient_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    frame = _sorted_feature_frame(obs_mask, feat_cols)
    frame["patient_id"] = frame["patient_id"].astype(str)
    if patient_ids is not None:
        frame = frame[frame["patient_id"].isin(set(str(pid) for pid in patient_ids))].copy()

    td_unit = pd.Timedelta(parse_bin_size(bin_size))
    out: dict[str, np.ndarray] = {}

    for pid, grp in frame.groupby("patient_id", sort=False):
        grp = grp.sort_values("bin_time", kind="stable")
        mask_vals = grp[feat_cols].to_numpy(dtype=np.float32)
        times = pd.to_datetime(grp["bin_time"])
        gaps = np.zeros(len(grp), dtype=np.float32)
        for t in range(1, len(grp)):
            gaps[t] = float((times.iloc[t] - times.iloc[t - 1]) / td_unit)

        delta = np.zeros_like(mask_vals, dtype=np.float32)
        if len(grp) > 1:
            for t in range(1, len(grp)):
                prev_obs = mask_vals[t - 1] > 0.5
                delta[t] = np.where(prev_obs, gaps[t], delta[t - 1] + gaps[t])
        out[str(pid)] = delta

    return out


def build_time_delta_tensor(
    time_delta_map: dict[str, np.ndarray],
    patient_ids: list[str],
    max_len: int,
    input_dim: int,
) -> torch.Tensor:
    out = np.zeros((len(patient_ids), max_len, input_dim), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        seq = time_delta_map.get(str(pid))
        if seq is None or max_len <= 0:
            continue
        if seq.shape[0] > max_len:
            seq = seq[-max_len:]
        out[i, : seq.shape[0], :] = seq
    return torch.from_numpy(out)


def build_group_sequence_tensor(
    *,
    binned: pd.DataFrame,
    groups: list[FeatureGroup],
    feat_cols: list[str],
    patient_ids: list[str] | None = None,
    max_len: int | None = None,
    reduce: str = "mean",
) -> torch.Tensor:
    if reduce not in {"mean", "max"}:
        raise ValueError(f"Unsupported reduce={reduce!r}")

    frame = _sorted_feature_frame(binned, feat_cols)
    seq_pids, seqs, _ = build_patient_sequences(frame, feat_cols)
    seq_map = {str(pid): seq.astype(np.float32, copy=False) for pid, seq in zip(seq_pids, seqs, strict=True)}
    aligned_pids = [str(pid) for pid in (patient_ids or list(seq_pids))]
    out_max_len = _infer_max_len(patient_ids=aligned_pids, seq_map=seq_map, max_len=max_len)

    out = np.zeros((len(aligned_pids), out_max_len, len(groups)), dtype=np.float32)
    for i, pid in enumerate(aligned_pids):
        seq = seq_map.get(pid)
        if seq is None or out_max_len <= 0:
            continue
        if seq.shape[0] > out_max_len:
            seq = seq[-out_max_len:]
        agg = np.zeros((seq.shape[0], len(groups)), dtype=np.float32)
        for g_idx, group in enumerate(groups):
            block = seq[:, group.indices]
            if reduce == "mean":
                agg[:, g_idx] = block.mean(axis=1)
            else:
                agg[:, g_idx] = block.max(axis=1)
        out[i, : agg.shape[0], :] = agg
    return torch.from_numpy(out)


def build_group_mask_tensor(
    *,
    obs_mask: pd.DataFrame,
    groups: list[FeatureGroup],
    feat_cols: list[str],
    patient_ids: list[str] | None = None,
    max_len: int | None = None,
) -> torch.Tensor:
    return build_group_sequence_tensor(
        binned=obs_mask,
        groups=groups,
        feat_cols=feat_cols,
        patient_ids=patient_ids,
        max_len=max_len,
        reduce="max",
    )
