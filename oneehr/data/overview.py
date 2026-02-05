from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneehr.config.schema import DatasetConfig


@dataclass(frozen=True)
class DatasetOverview:
    n_events: int
    n_patients: int
    time_min: str | None
    time_max: str | None
    label_summary: dict[str, float] | None
    events_per_patient: dict[str, float]
    top_codes: list[dict[str, object]]


def build_dataset_overview(events: pd.DataFrame, cfg: DatasetConfig, *, top_k_codes: int = 20) -> DatasetOverview:
    pid = cfg.patient_id_col
    tcol = cfg.time_col
    code = cfg.code_col
    label = cfg.label_col

    if pid not in events.columns or tcol not in events.columns or code not in events.columns:
        raise ValueError(f"events missing required columns for overview: {[pid, tcol, code]}")

    df = events.copy()
    df[pid] = df[pid].astype(str)

    n_events = int(len(df))
    n_patients = int(df[pid].nunique())

    time_min = None
    time_max = None
    if tcol in df.columns:
        tt = pd.to_datetime(df[tcol], errors="coerce")
        if tt.notna().any():
            time_min = str(tt.min())
            time_max = str(tt.max())

    label_summary = None
    if label in df.columns:
        # Label is repeated per-event; summarize per patient.
        lp = (
            df[[pid, label]]
            .dropna(subset=[label])
            .drop_duplicates(subset=[pid], keep="last")
        )
        if not lp.empty:
            y = pd.to_numeric(lp[label], errors="coerce").dropna()
            if not y.empty:
                label_summary = {
                    "n_labeled_patients": float(len(y)),
                    "mean": float(y.mean()),
                    "std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                    "min": float(y.min()),
                    "p25": float(np.percentile(y, 25)),
                    "median": float(np.percentile(y, 50)),
                    "p75": float(np.percentile(y, 75)),
                    "max": float(y.max()),
                }
                uniq = sorted(y.unique().tolist())
                if len(uniq) <= 5:
                    label_summary["unique_values"] = uniq
                    if set(uniq).issubset({0.0, 1.0}):
                        label_summary["positive_rate"] = float((y == 1.0).mean())

    cnt = df.groupby(pid, sort=False).size()
    events_per_patient = {
        "mean": float(cnt.mean()) if len(cnt) else 0.0,
        "std": float(cnt.std(ddof=1)) if len(cnt) > 1 else 0.0,
        "min": float(cnt.min()) if len(cnt) else 0.0,
        "p25": float(np.percentile(cnt, 25)) if len(cnt) else 0.0,
        "median": float(np.percentile(cnt, 50)) if len(cnt) else 0.0,
        "p75": float(np.percentile(cnt, 75)) if len(cnt) else 0.0,
        "max": float(cnt.max()) if len(cnt) else 0.0,
    }

    vc = df[code].astype(str).value_counts().head(int(top_k_codes))
    top_codes = [{"code": str(c), "count": int(n)} for c, n in vc.items()]

    return DatasetOverview(
        n_events=n_events,
        n_patients=n_patients,
        time_min=time_min,
        time_max=time_max,
        label_summary=label_summary,
        events_per_patient=events_per_patient,
        top_codes=top_codes,
    )

