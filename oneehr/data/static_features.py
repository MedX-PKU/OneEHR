from __future__ import annotations

import pandas as pd

from oneehr.config.schema import DynamicTableConfig, StaticFeaturesConfig


def build_static_features(
    events: pd.DataFrame,
    dataset: DynamicTableConfig,
    cfg: StaticFeaturesConfig,
) -> pd.DataFrame | None:
    """Build patient-level static features from the raw event table.

    Design:
    - Source is the same raw event table (doctor-friendly single table input).
    - Users select which columns should be treated as static via config.
    - We aggregate per patient (first/last by time_col) to get a single row per patient.

    Returns:
    - DataFrame indexed by patient_id (string) or None if disabled.
    """

    if not cfg.enabled:
        return None
    if not cfg.cols:
        return None

    pid_col = dataset.patient_id_col
    time_col = dataset.time_col
    missing = [c for c in [pid_col, time_col, *cfg.cols] if c not in events.columns]
    if missing:
        raise ValueError(f"events missing columns required for static features: {missing}")

    df = events[[pid_col, time_col, *cfg.cols]].copy()
    df = df.rename(columns={pid_col: "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.sort_values(["patient_id", time_col], kind="stable")

    if cfg.agg not in {"first", "last"}:
        raise ValueError(f"Unsupported static_features.agg={cfg.agg!r}")

    if cfg.agg == "first":
        out = df.groupby("patient_id", sort=False).first(numeric_only=False)
    else:
        out = df.groupby("patient_id", sort=False).last(numeric_only=False)

    # Keep only selected static columns.
    out = out[cfg.cols]

    # Basic one-hot / numeric handling is deferred to existing postprocess pipeline;
    # here we keep raw columns and let tabular postprocess fit on train split.
    return out
