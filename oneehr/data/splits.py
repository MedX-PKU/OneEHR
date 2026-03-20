from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.config.schema import SplitConfig


def make_patient_index(events: pd.DataFrame, time_col: str, patient_id_col: str) -> pd.DataFrame:
    df = events[[patient_id_col, time_col]].copy()
    df[patient_id_col] = df[patient_id_col].astype(str)
    df[time_col] = pd.to_datetime(df[time_col], errors="raise")
    g = df.groupby(patient_id_col, sort=False)[time_col]
    out = pd.DataFrame(
        {
            "patient_id": g.min().index.astype(str),
            "min_time": g.min().to_numpy(),
            "max_time": g.max().to_numpy(),
        }
    )
    return out


def make_patient_index_from_static(static: pd.DataFrame, patient_id_col: str = "patient_id") -> pd.DataFrame:
    """Create a patient index for static-only datasets."""
    if patient_id_col not in static.columns:
        raise ValueError(f"static missing required column: {patient_id_col!r}")
    pids = static[patient_id_col].astype(str).dropna().unique()
    out = pd.DataFrame({"patient_id": pids})
    out["min_time"] = pd.NaT
    out["max_time"] = pd.NaT
    return out


@dataclass(frozen=True)
class Split:
    name: str
    train_patients: np.ndarray
    val_patients: np.ndarray
    test_patients: np.ndarray


def expand_splits_for_repeat(splits: list[Split], repeat: int) -> list[Split]:
    if int(repeat) <= 1:
        return list(splits)
    expanded: list[Split] = []
    for sp in splits:
        for repeat_idx in range(int(repeat)):
            expanded.append(
                Split(
                    name=f"{sp.name}__r{repeat_idx}",
                    train_patients=sp.train_patients,
                    val_patients=sp.val_patients,
                    test_patients=sp.test_patients,
                )
            )
    return expanded


def save_splits(splits: list[Split], out_dir: Path) -> None:
    """Write each Split as {name}.json with train/val/test patient arrays."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        payload = {
            "name": sp.name,
            "train_patients": sp.train_patients.tolist(),
            "val_patients": sp.val_patients.tolist(),
            "test_patients": sp.test_patients.tolist(),
        }
        (out_dir / f"{sp.name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def load_splits(split_dir: Path) -> list[Split]:
    """Read back into list[Split], sorted by filename."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    splits: list[Split] = []
    for path in sorted(split_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        splits.append(
            Split(
                name=str(data["name"]),
                train_patients=np.array(data["train_patients"], dtype=str),
                val_patients=np.array(data["val_patients"], dtype=str),
                test_patients=np.array(data["test_patients"], dtype=str),
            )
        )
    return splits


def require_saved_splits(split_dir: Path, *, context: str) -> list[Split]:
    split_dir = Path(split_dir)
    splits = load_splits(split_dir)
    if splits:
        return splits
    raise SystemExit(
        f"Missing saved splits at {split_dir}. Run `oneehr preprocess` first before {context}."
    )


_REPEAT_RE = re.compile(r"__r(\d+)$")


def _parse_repeat_index(name: str) -> int:
    """Extract repeat index from split names like ``fold0__r2`` → 2.

    Returns 0 for non-repeat names.
    """
    m = _REPEAT_RE.search(name)
    return int(m.group(1)) if m else 0


def make_splits(
    patient_index: pd.DataFrame,
    split: SplitConfig,
) -> list[Split]:
    """Generate patient-level splits.

    patient_index must contain at least:
    - patient_id
    - min_time
    - max_time
    """

    if "patient_id" not in patient_index.columns:
        raise ValueError("patient_index missing 'patient_id'")

    patients = patient_index["patient_id"].astype(str).unique()
    rng = np.random.default_rng(split.seed)

    kind = split.kind.lower()
    if kind == "kfold":
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=split.n_splits)
        # Use dummy X, groups=patient_id ensures grouping.
        X = np.zeros((len(patients), 1))
        groups = patients
        splits: list[Split] = []
        # Note: GroupKFold's splitting is deterministic given the order of
        # `patients` and does not use `random_state`. We shuffle patient order
        # here to respect `split.seed` while preserving patient-level grouping.
        patients = patients[rng.permutation(len(patients))]
        groups = patients
        X = np.zeros((len(patients), 1))

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, groups=groups), start=0):
            train_patients = patients[train_idx]
            test_patients = patients[test_idx]
            # Create validation by sampling from train.
            if split.val_size > 0:
                n_val = max(1, int(round(len(train_patients) * split.val_size)))
                perm = rng.permutation(len(train_patients))
                val_patients = train_patients[perm[:n_val]]
                train_patients = train_patients[perm[n_val:]]
            else:
                val_patients = np.array([], dtype=str)
            splits.append(
                Split(
                    name=f"fold{fold}",
                    train_patients=train_patients,
                    val_patients=val_patients,
                    test_patients=test_patients,
                )
            )
        if split.fold_index is None:
            return splits
        if split.fold_index < 0 or split.fold_index >= len(splits):
            raise ValueError(
                f"split.fold_index out of range: {split.fold_index}. "
                f"Expected 0..{len(splits) - 1}"
            )
        return [splits[int(split.fold_index)]]

    if kind == "random":
        perm = rng.permutation(len(patients))
        patients = patients[perm]
        n_test = max(1, int(round(len(patients) * split.test_size)))
        test_patients = patients[:n_test]
        rem = patients[n_test:]

        n_val = max(1, int(round(len(rem) * split.val_size))) if split.val_size > 0 else 0
        val_patients = rem[:n_val]
        train_patients = rem[n_val:]
        return [
            Split(
                name="split0",
                train_patients=train_patients,
                val_patients=val_patients,
                test_patients=test_patients,
            )
        ]

    if kind == "time":
        if split.time_boundary is None:
            raise ValueError("split.time requires split.time_boundary")
        boundary = pd.to_datetime(split.time_boundary, errors="raise")
        if "max_time" not in patient_index.columns:
            raise ValueError("patient_index missing 'max_time' for time split")

        pid = patient_index[["patient_id", "max_time"]].copy()
        pid["patient_id"] = pid["patient_id"].astype(str)
        pre = pid[pid["max_time"] < boundary]["patient_id"].to_numpy()
        post = pid[pid["max_time"] >= boundary]["patient_id"].to_numpy()
        pre = pre.astype(str)
        post = post.astype(str)

        # Default time split: pre-boundary train/val, post-boundary test.
        rng.shuffle(pre)
        if split.val_size > 0 and len(pre) > 1:
            n_val = max(1, int(round(len(pre) * split.val_size)))
            val = pre[:n_val]
            train = pre[n_val:]
        else:
            val = np.array([], dtype=str)
            train = pre
        return [Split(name="time0", train_patients=train, val_patients=val, test_patients=post)]

    raise ValueError(f"Unsupported split.kind={split.kind!r}. Expected kfold|random|time")


def build_splits_for_dataset(
    *,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    split: SplitConfig,
    repeat: int = 1,
) -> list[Split]:
    if dynamic is not None:
        patient_index = make_patient_index(dynamic, "event_time", "patient_id")
    elif static is not None:
        patient_index = make_patient_index_from_static(static, patient_id_col="patient_id")
    else:
        raise SystemExit("dataset.dynamic or dataset.static is required to materialize splits.")

    splits = make_splits(patient_index, split)
    return expand_splits_for_repeat(splits, repeat)
