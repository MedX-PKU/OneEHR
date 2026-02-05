from __future__ import annotations

import pandas as pd


def build_labels(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg,
) -> pd.DataFrame:
    """Example label_fn (new signature).

    Demonstrates N-N (time) label generation by combining inputs:
    - `dynamic`: used to find each patient's last observed time
    - `label`: optional task-agnostic label event table (long format)

    Output columns for N-N:
    - patient_id
    - label_time (OneEHR will floor to bin_time if needed)
    - label
    - mask
    """

    _ = static  # not used in this example

    # Find last observed dynamic time per patient.
    df = dynamic[["patient_id", "event_time"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="raise")
    df = df.sort_values(["patient_id", "event_time"], kind="stable")
    last_time = df.groupby("patient_id", sort=False).tail(1).rename(columns={"event_time": "label_time"})

    # If label.csv exists, pick one label_code for the task; else synthesize a dummy label.
    if label is not None and not label.empty:
        lab = label[["patient_id", "label_time", "label_code", "label_value"]].copy()
        lab["patient_id"] = lab["patient_id"].astype(str)
        lab["label_time"] = pd.to_datetime(lab["label_time"], errors="raise")

        # Example rule: use label_code="outcome" when binary, else "los".
        wanted = "outcome" if cfg.task.kind == "binary" else "los"
        lab = lab.loc[lab["label_code"].astype(str) == wanted].copy()
        if lab.empty:
            raise ValueError(f"label.csv has no rows with label_code={wanted!r}")

        # For each patient, take the last label at/before the last observed time.
        merged = last_time.merge(lab, on="patient_id", how="left")
        merged = merged.loc[merged["label_time_y"] <= merged["label_time_x"]].copy()
        merged = merged.sort_values(["patient_id", "label_time_y"], kind="stable").groupby("patient_id", sort=False).tail(1)
        out = merged.rename(columns={"label_value": "label", "label_time_x": "label_time"})[
            ["patient_id", "label_time", "label"]
        ].copy()
    else:
        out = last_time.copy()
        out["label"] = 1

    out["mask"] = 1
    return out[["patient_id", "label_time", "label", "mask"]]
