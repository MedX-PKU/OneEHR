from __future__ import annotations

import numpy as np
import pandas as pd

from oneehr.config.schema import DynamicTableConfig


def build_dataset_overview(
    dynamic: pd.DataFrame,
    cfg: DynamicTableConfig,
    *,
    top_k_codes: int = 20,
) -> dict[str, object]:
    _ = cfg
    pid = "patient_id"
    tcol = "event_time"
    code = "code"

    if pid not in dynamic.columns or tcol not in dynamic.columns or code not in dynamic.columns:
        raise ValueError(f"dynamic.csv missing required columns for overview: {[pid, tcol, code]}")

    df = dynamic.copy()
    df[pid] = df[pid].astype(str)

    out: dict[str, object] = {
        "n_events": int(len(df)),
        "n_patients": int(df[pid].nunique()),
    }

    tt = pd.to_datetime(df[tcol], errors="coerce")
    if tt.notna().any():
        out["time_min"] = str(tt.min())
        out["time_max"] = str(tt.max())

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
