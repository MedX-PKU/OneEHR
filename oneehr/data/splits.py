from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.config.schema import SplitConfig


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_patient_index(events: pd.DataFrame) -> pd.DataFrame:
    df = events[["patient_id", "event_time"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="raise")
    g = df.groupby("patient_id", sort=False)["event_time"]
    return pd.DataFrame({
        "patient_id": g.min().index.astype(str),
        "min_time": g.min().to_numpy(),
        "max_time": g.max().to_numpy(),
    })


def make_split(patient_index: pd.DataFrame, cfg: SplitConfig) -> Split:
    patients = patient_index["patient_id"].astype(str).unique()
    rng = np.random.default_rng(cfg.seed)
    kind = cfg.kind.lower()

    if kind == "random":
        perm = rng.permutation(len(patients))
        patients = patients[perm]
        n_test = max(1, int(round(len(patients) * cfg.test_size)))
        test = patients[:n_test]
        rem = patients[n_test:]
        n_val = max(1, int(round(len(rem) * cfg.val_size))) if cfg.val_size > 0 else 0
        val = rem[:n_val]
        train = rem[n_val:]
        return Split(train=train, val=val, test=test)

    if kind == "time":
        if cfg.time_boundary is None:
            raise ValueError("split.kind='time' requires split.time_boundary")
        boundary = pd.to_datetime(cfg.time_boundary, errors="raise")
        if "max_time" not in patient_index.columns:
            raise ValueError("patient_index missing 'max_time' for time split")
        pid = patient_index[["patient_id", "max_time"]].copy()
        pid["patient_id"] = pid["patient_id"].astype(str)
        pre = pid[pid["max_time"] < boundary]["patient_id"].to_numpy().astype(str)
        post = pid[pid["max_time"] >= boundary]["patient_id"].to_numpy().astype(str)
        rng.shuffle(pre)
        if cfg.val_size > 0 and len(pre) > 1:
            n_val = max(1, int(round(len(pre) * cfg.val_size)))
            val = pre[:n_val]
            train = pre[n_val:]
        else:
            val = np.array([], dtype=str)
            train = pre
        return Split(train=train, val=val, test=post)

    raise ValueError(f"Unsupported split.kind={cfg.kind!r}. Expected random|time")


def save_split(split: Split, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": split.train.tolist(),
        "val": split.val.tolist(),
        "test": split.test.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: Path) -> Split:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Split(
        train=np.array(data["train"], dtype=str),
        val=np.array(data["val"], dtype=str),
        test=np.array(data["test"], dtype=str),
    )


def require_split(path: Path, *, context: str = "") -> Split:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Missing split.json at {path}. Run `oneehr preprocess` first"
            + (f" before {context}." if context else ".")
        )
    return load_split(path)
