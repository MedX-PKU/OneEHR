from __future__ import annotations

import pandas as pd


def validate_patient_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "label"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"labels.parquet missing columns: {missing}")
    out = df[["patient_id", "label"]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    return out


def validate_time_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "bin_time", "label"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"labels.parquet missing columns: {missing}")
    cols = ["patient_id", "bin_time", "label"]
    if "mask" in df.columns:
        cols.append("mask")
    out = df[cols].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    return out

