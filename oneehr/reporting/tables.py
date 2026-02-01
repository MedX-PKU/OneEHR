from __future__ import annotations

import pandas as pd


def summarize_metrics(metrics_per_split: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table with mean/std over splits.

    Expects a DataFrame like:
      split, auroc, auprc, ...
    """

    if "split" not in metrics_per_split.columns:
        raise ValueError("metrics_per_split must contain 'split'")

    metric_cols = [c for c in metrics_per_split.columns if c != "split"]
    rows = []
    for col in metric_cols:
        s = metrics_per_split[col].astype(float)
        rows.append({"metric": col, "mean": float(s.mean()), "std": float(s.std(ddof=1))})
    return pd.DataFrame(rows).sort_values("metric")

