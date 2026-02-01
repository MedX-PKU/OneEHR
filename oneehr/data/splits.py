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
        return splits

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
        train = pid[pid["max_time"] < boundary]["patient_id"].to_numpy()
        test = pid[pid["max_time"] >= boundary]["patient_id"].to_numpy()
        rng.shuffle(train)
        if split.val_size > 0 and len(train) > 1:
            n_val = max(1, int(round(len(train) * split.val_size)))
            val = train[:n_val]
            train = train[n_val:]
        else:
            val = np.array([], dtype=str)
        return [Split(name="time0", train_patients=train, val_patients=val, test_patients=test)]

    raise ValueError(f"Unsupported split.kind={split.kind!r}. Expected kfold|random|time")

