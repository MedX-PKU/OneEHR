from __future__ import annotations

import pandas as pd


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
    """Create a patient index for static-only datasets.

    min_time/max_time are set to NaT because there is no longitudinal timeline.
    This still allows patient-level grouped splits.
    """

    if patient_id_col not in static.columns:
        raise ValueError(f"static missing required column: {patient_id_col!r}")
    pids = static[patient_id_col].astype(str).dropna().unique()
    out = pd.DataFrame({"patient_id": pids})
    out["min_time"] = pd.NaT
    out["max_time"] = pd.NaT
    return out
