from __future__ import annotations

import numpy as np
import pandas as pd

from oneehr.config.schema import DatasetConfig


def build_dataset_overview(events: pd.DataFrame, cfg: DatasetConfig, *, top_k_codes: int = 20) -> dict[str, object]:
    pid = cfg.patient_id_col
    tcol = cfg.time_col
    code = cfg.code_col
    label = cfg.label_col

    if pid not in events.columns or tcol not in events.columns or code not in events.columns:
        raise ValueError(f"events missing required columns for overview: {[pid, tcol, code]}")

    df = events.copy()
    df[pid] = df[pid].astype(str)

    out: dict[str, object] = {
        "n_events": int(len(df)),
        "n_patients": int(df[pid].nunique()),
    }

    tt = pd.to_datetime(df[tcol], errors="coerce")
    if tt.notna().any():
        out["time_min"] = str(tt.min())
        out["time_max"] = str(tt.max())

    if label in df.columns:
        lp = (
            df[[pid, label]]
            .dropna(subset=[label])
            .drop_duplicates(subset=[pid], keep="last")
        )
        if not lp.empty:
            y = pd.to_numeric(lp[label], errors="coerce").dropna()
            if not y.empty:
                out["label"] = {
                    "n_labeled_patients": int(len(y)),
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
                    out["label"]["unique_values"] = uniq
                    if set(uniq).issubset({0.0, 1.0}):
                        out["label"]["positive_rate"] = float((y == 1.0).mean())

    cnt = df.groupby(pid, sort=False).size()
    out["events_per_patient"] = {
        "mean": float(cnt.mean()) if len(cnt) else 0.0,
        "std": float(cnt.std(ddof=1)) if len(cnt) > 1 else 0.0,
        "min": int(cnt.min()) if len(cnt) else 0,
        "p25": float(np.percentile(cnt, 25)) if len(cnt) else 0.0,
        "median": float(np.percentile(cnt, 50)) if len(cnt) else 0.0,
        "p75": float(np.percentile(cnt, 75)) if len(cnt) else 0.0,
        "max": int(cnt.max()) if len(cnt) else 0,
    }

    vc = df[code].astype(str).value_counts().head(int(top_k_codes))
    out["top_codes"] = [{"code": str(c), "count": int(n)} for c, n in vc.items()]
    return out

