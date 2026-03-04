from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.config.schema import SplitConfig


@dataclass(frozen=True)
class Split:
    name: str
    train_patients: np.ndarray
    val_patients: np.ndarray
    test_patients: np.ndarray


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

        # Nested CV: fixed random test set, CV folds on remainder.
        if split.inner_kind is not None:
            inner_kind = split.inner_kind.lower()
            if inner_kind != "kfold":
                raise ValueError("split.random currently supports inner_kind='kfold' only")
            n_splits = split.inner_n_splits or split.n_splits
            from sklearn.model_selection import GroupKFold

            gkf = GroupKFold(n_splits=int(n_splits))
            X0 = np.zeros((len(rem), 1))
            groups0 = rem
            out: list[Split] = []
            for fold, (train_idx, val_idx) in enumerate(gkf.split(X0, groups=groups0), start=0):
                tr = rem[train_idx]
                va = rem[val_idx]
                out.append(Split(name=f"random0_fold{fold}", train_patients=tr, val_patients=va, test_patients=test_patients))
            if split.fold_index is None:
                return out
            if split.fold_index < 0 or split.fold_index >= len(out):
                raise ValueError(
                    f"split.fold_index out of range: {split.fold_index}. "
                    f"Expected 0..{len(out) - 1}"
                )
            return [out[int(split.fold_index)]]

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

        # Nested CV: fixed prospective test (post-boundary), and CV splits on pre-boundary pool.
        if split.inner_kind is not None:
            inner_kind = split.inner_kind.lower()
            if inner_kind != "kfold":
                raise ValueError("split.time currently supports inner_kind='kfold' only")
            n_splits = split.inner_n_splits or split.n_splits
            from sklearn.model_selection import GroupKFold

            gkf = GroupKFold(n_splits=int(n_splits))
            X0 = np.zeros((len(pre), 1))
            groups0 = pre
            out: list[Split] = []
            for fold, (train_idx, val_idx) in enumerate(gkf.split(X0, groups=groups0), start=0):
                tr = pre[train_idx]
                va = pre[val_idx]
                out.append(Split(name=f"time0_fold{fold}", train_patients=tr, val_patients=va, test_patients=post))
            if split.fold_index is None:
                return out
            if split.fold_index < 0 or split.fold_index >= len(out):
                raise ValueError(
                    f"split.fold_index out of range: {split.fold_index}. "
                    f"Expected 0..{len(out) - 1}"
                )
            return [out[int(split.fold_index)]]

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
