from __future__ import annotations

import pandas as pd


def build_labels(dynamic: pd.DataFrame, static: pd.DataFrame | None, label: pd.DataFrame | None, cfg) -> pd.DataFrame:
    """TJH example label_fn selecting task from label.csv.

    - binary -> label_code="outcome"
    - regression -> label_code="los"

    Returns patient-level labels: columns `patient_id`, `label`.
    """

    _ = dynamic, static
    if label is None or label.empty:
        raise ValueError("TJH example requires label.csv (dataset.label.path).")

    df = label.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["label_time"] = pd.to_datetime(df["label_time"], errors="raise")

    wanted = "outcome" if cfg.task.kind == "binary" else "los"
    df = df.loc[df["label_code"].astype(str) == wanted].copy()
    if df.empty:
        raise ValueError(f"label.csv has no rows with label_code={wanted!r}")

    df = df.sort_values(["patient_id", "label_time"], kind="stable")
    last = df.groupby("patient_id", sort=False).tail(1)
    out = last.rename(columns={"label_value": "label"})[["patient_id", "label"]].copy()
    return out

