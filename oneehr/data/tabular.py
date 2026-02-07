from __future__ import annotations

import pandas as pd


def make_patient_tabular(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert binned long table into one-row-per-patient tabular features.

    Takes features from the last time bin (discretized) for each patient.
    """

    if "patient_id" not in binned.columns:
        raise ValueError("binned table must contain patient_id")
    if "label" not in binned.columns:
        # Labels are optional at preprocess time; return empty y.
        feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
        if not feature_cols:
            raise ValueError("No feature columns found (expected num__/cat__ prefix)")
        binned = binned.sort_values(["patient_id", "bin_time"], kind="stable")
        X = binned.groupby("patient_id", sort=False)[feature_cols].last()
        y = pd.Series([], dtype=float, name="label")
        return X, y

    feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected num__/cat__ prefix)")

    binned = binned.sort_values(["patient_id", "bin_time"], kind="stable")
    X = binned.groupby("patient_id", sort=False)[feature_cols].last()
    y = binned.groupby("patient_id", sort=False)["label"].last()
    return X, y


def make_time_tabular(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return one-row-per-(patient_id, bin_time) tabular features.

    Returns:
    - X: features indexed by a RangeIndex
    - y: label aligned to X
    - key: DataFrame with columns patient_id, bin_time (aligned to X)
    """

    required = {"patient_id", "bin_time"}
    missing = [c for c in required if c not in binned.columns]
    if missing:
        raise ValueError(f"binned missing columns: {missing}")

    feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected num__/cat__ prefix)")

    cols = ["patient_id", "bin_time", *feature_cols]
    if "label" in binned.columns:
        cols.insert(2, "label")
    df = binned[cols].copy().sort_values(["patient_id", "bin_time"], kind="stable")
    key = df[["patient_id", "bin_time"]].reset_index(drop=True)
    X = df[feature_cols].reset_index(drop=True)
    if "label" in df.columns:
        df = df.dropna(subset=["label"])
        key = df[["patient_id", "bin_time"]].reset_index(drop=True)
        X = df[feature_cols].reset_index(drop=True)
        y = df["label"].reset_index(drop=True)
    else:
        y = pd.Series([], dtype=float, name="label")
    return X, y, key
