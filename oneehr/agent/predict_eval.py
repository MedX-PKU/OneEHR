from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_prediction_rows(df: pd.DataFrame, *, task_kind: str) -> dict[str, Any]:
    total_rows = int(len(df))
    parsed_ok = int(df.get("parsed_ok", pd.Series(dtype=bool)).fillna(False).sum()) if total_rows else 0
    grounded = df[df["ground_truth"].notna()].copy() if "ground_truth" in df.columns else df.iloc[0:0].copy()
    scored = grounded[grounded["parsed_ok"].fillna(False)].copy() if not grounded.empty else grounded

    metrics: dict[str, Any] = {}
    if not scored.empty:
        y_true = scored["ground_truth"].to_numpy(dtype=float)
        if task_kind == "binary":
            from oneehr.eval.metrics import binary_metrics

            y_score = scored["probability"].to_numpy(dtype=float)
            metrics = dict(binary_metrics(y_true, y_score).metrics)
        elif task_kind == "regression":
            from oneehr.eval.metrics import regression_metrics

            y_pred = scored["prediction"].to_numpy(dtype=float)
            metrics = dict(regression_metrics(y_true, y_pred).metrics)
        else:
            raise ValueError(f"Unsupported agent task kind: {task_kind!r}")

    token_cols = [
        "token_usage_prompt",
        "token_usage_completion",
        "token_usage_total",
        "latency_ms",
    ]
    perf: dict[str, Any] = {}
    for col in token_cols:
        if col in df.columns and df[col].notna().any():
            perf[f"mean_{col}"] = float(np.nanmean(df[col].to_numpy(dtype=float)))

    return {
        "total_rows": total_rows,
        "parsed_ok_rows": parsed_ok,
        "parse_success_rate": (float(parsed_ok) / float(total_rows)) if total_rows else 0.0,
        "ground_truth_rows": int(len(grounded)),
        "scored_rows": int(len(scored)),
        "coverage": (float(len(scored)) / float(total_rows)) if total_rows else 0.0,
        "metrics": metrics,
        **perf,
    }
