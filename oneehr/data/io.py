from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dynamic_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"dynamic must be a .csv file. Got: {path}")
    df = pd.read_csv(path).copy()

    required = ["patient_id", "event_time", "code", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"dynamic.csv missing required columns: {missing}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="raise")
    df["code"] = df["code"].astype(str)
    return df[required]


def load_dynamic_table_optional(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return load_dynamic_table(path)


def load_static_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"static must be a .csv file. Got: {path}")
    df = pd.read_csv(path).copy()
    if "patient_id" not in df.columns:
        raise ValueError("static.csv missing required column: 'patient_id'")
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def load_label_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"label must be a .csv file. Got: {path}")
    df = pd.read_csv(path).copy()
    required = ["patient_id", "label_time", "label_code", "label_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"label table missing columns: {missing}")
    df["patient_id"] = df["patient_id"].astype(str)
    df["label_time"] = pd.to_datetime(df["label_time"], errors="raise")
    df["label_code"] = df["label_code"].astype(str)
    return df[required]
