from __future__ import annotations

import pandas as pd


def build_labels(events: pd.DataFrame, cfg) -> pd.DataFrame:
    """Example label_fn.

    This example returns per-bin (N-N) labels with a mask.

    - For each patient, we create a label only for the LAST observed event_time.
    - The label is copied from the patient's existing `label` column.
    - OneEHR will convert `label_time` -> internal `bin_time` based on cfg.preprocess.bin_size.

    Required output columns for N-N:
    - patient_id
    - label_time (or bin_time if you set labels.bin_from_time_col=false)
    - label
    - mask (1=keep, 0=ignore)
    """

    required = {"patient_id", "event_time", "label"}
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"events missing columns: {missing}")

    df = events[["patient_id", "event_time", "label"]].copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="raise")
    df = df.dropna(subset=["label"])
    df = df.sort_values(["patient_id", "event_time"], kind="stable")

    last = df.groupby("patient_id", sort=False).tail(1)
    out = last.rename(columns={"event_time": "label_time"})
    out["mask"] = 1
    return out[["patient_id", "label_time", "label", "mask"]]

