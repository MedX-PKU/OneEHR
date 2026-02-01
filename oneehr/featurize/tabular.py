from __future__ import annotations

import pandas as pd


def make_patient_tabular(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert binned long table into one-row-per-patient tabular features.

    Current behavior (MVP): aggregate each feature column over time with mean.
    """

    if "patient_id" not in binned.columns or "label" not in binned.columns:
        raise ValueError("binned table must contain patient_id and label")

    feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected num__/cat__ prefix)")

    X = binned.groupby("patient_id", sort=False)[feature_cols].mean()
    y = binned.groupby("patient_id", sort=False)["label"].last()
    return X, y


def make_time_tabular(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return one-row-per-(patient_id, bin_time) tabular features.

    Returns:
    - X: features indexed by a RangeIndex
    - y: label aligned to X
    - key: DataFrame with columns patient_id, bin_time (aligned to X)
    """

    required = {"patient_id", "bin_time", "label"}
    missing = [c for c in required if c not in binned.columns]
    if missing:
        raise ValueError(f"binned missing columns: {missing}")

    feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected num__/cat__ prefix)")

    df = binned[["patient_id", "bin_time", "label", *feature_cols]].copy()
    df = df.dropna(subset=["label"]).sort_values(["patient_id", "bin_time"], kind="stable")
    key = df[["patient_id", "bin_time"]].reset_index(drop=True)
    X = df[feature_cols].reset_index(drop=True)
    y = df["label"].reset_index(drop=True)
    return X, y, key
