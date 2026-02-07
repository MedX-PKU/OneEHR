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
    del cfg
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


def build_feature_overview(
    *,
    dynamic_feature_columns: list[str],
    static_feature_columns: list[str] | None = None,
    top_k_categoricals: int = 30,
) -> dict[str, object]:
    """Build a clinician-friendly summary of the preprocessed feature space.

    This is intentionally *display-oriented* and should not be used as the source
    of truth for training. It exists to help doctor-facing users understand what
    the pipeline produced.
    """

    def _summarize(cols: list[str]) -> dict[str, object]:
        num = [c for c in cols if c.startswith("num__")]
        cat = [c for c in cols if c.startswith("cat__")]

        # Display names:
        # - num__Age -> Age
        # - cat__Sex__M -> Sex=M
        def _disp(c: str) -> str:
            if c.startswith("num__"):
                return c[len("num__") :]
            if c.startswith("cat__"):
                rest = c[len("cat__") :]
                if "__" in rest:
                    left, right = rest.split("__", 1)
                    return f"{left}={right}"
                return rest
            return c

        # Group categoricals by base name (before the first __).
        cat_groups: dict[str, int] = {}
        for c in cat:
            rest = c[len("cat__") :]
            base = rest.split("__", 1)[0] if "__" in rest else rest
            cat_groups[base] = cat_groups.get(base, 0) + 1

        top_cat = sorted(cat_groups.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k_categoricals)]

        return {
            "n_total": int(len(cols)),
            "n_numeric": int(len(num)),
            "n_categorical": int(len(cat)),
            "numeric_features": [_disp(c) for c in num],
            "categorical_feature_bases": [{"name": k, "n_levels": int(v)} for k, v in top_cat],
        }

    out: dict[str, object] = {"dynamic": _summarize(dynamic_feature_columns)}
    if static_feature_columns is not None:
        out["static"] = _summarize(static_feature_columns)
    return out
