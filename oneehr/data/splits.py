from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneehr.config.schema import SplitConfig


@dataclass(frozen=True)
class Split:
    name: str
    train_patients: np.ndarray
    val_patients: np.ndarray
    test_patients: np.ndarray


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
