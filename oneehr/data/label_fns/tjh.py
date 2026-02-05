from __future__ import annotations

import pandas as pd


def build_outcome_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Patient-level Outcome labels for TJH.

    Assumes TJH loader keeps an `Outcome` column in the raw-turned-event table.
    Returns: DataFrame with columns [patient_id, label]
    """

    if "patient_id" not in events.columns or "Outcome" not in events.columns:
        raise ValueError("TJH outcome label_fn requires columns: patient_id, Outcome")
    df = events[["patient_id", "Outcome"]].dropna(subset=["Outcome"]).copy()
    df["patient_id"] = df["patient_id"].astype(str)
    # One label per patient
    out = df.drop_duplicates(subset=["patient_id"], keep="last").rename(columns={"Outcome": "label"})
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"])
    return out[["patient_id", "label"]]


def build_los_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Patient-level LOS labels for TJH.

    LOS computed from DischargeTime - RecordTime (days) using the first RecordTime.
    Returns: DataFrame with columns [patient_id, label]
    """

    req = {"patient_id", "event_time", "DischargeTime"}
    missing = [c for c in req if c not in events.columns]
    if missing:
        raise ValueError(f"TJH LOS label_fn requires columns: {sorted(req)}. Missing: {missing}")

    df = events[["patient_id", "event_time", "DischargeTime"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["DischargeTime"] = pd.to_datetime(df["DischargeTime"], errors="coerce")
    df = df.dropna(subset=["event_time", "DischargeTime"])

    # Use earliest event_time as baseline (matches legacy using RecordTime).
    base = (
        df.sort_values(["patient_id", "event_time"], kind="stable")
        .groupby("patient_id", sort=False)
        .first()
    )
    # Use date-level LOS (match legacy: DischargeTime and RecordTime were truncated to YYYY-MM-DD).
    los = (base["DischargeTime"].dt.normalize() - base["event_time"].dt.normalize()).dt.days
    # TJH convention in legacy script: negative values set to 0.
    los = los.clip(lower=0)
    out = los.rename("label").reset_index()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"])
    return out[["patient_id", "label"]]
