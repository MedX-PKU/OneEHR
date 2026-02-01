from __future__ import annotations

import numpy as np
import pandas as pd


def _ci95_mean(values: pd.Series) -> tuple[float, float]:
    """Normal-approx 95% CI for the mean: mean ± 1.96 * (std / sqrt(n))."""

    v = values.dropna().astype(float)
    n = int(v.shape[0])
    if n == 0:
        return float("nan"), float("nan")
    mean = float(v.mean())
    if n == 1:
        return mean, mean
    std = float(v.std(ddof=1))
    half = 1.96 * std / (n**0.5)
    return mean - half, mean + half


def _bootstrap_ci95_mean(values: pd.Series, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    v = values.dropna().astype(float).to_numpy()
    n = int(v.shape[0])
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_metrics(metrics_per_split: pd.DataFrame) -> pd.DataFrame:
    """Create a compact long table with mean/std/95% CI over splits."""

    if "split" not in metrics_per_split.columns:
        raise ValueError("metrics_per_split must contain 'split'")

    metric_cols = [c for c in metrics_per_split.columns if c != "split"]
    rows = []
    for col in metric_cols:
        s = metrics_per_split[col].astype(float)
        ci_low, ci_high = _bootstrap_ci95_mean(s)
        rows.append(
            {
                "metric": col,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "n": int(s.dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("metric")


def to_paper_wide_table(metrics_per_split: pd.DataFrame) -> pd.DataFrame:
    """Return a single-row wide table more like paper tables.

    Output columns look like:
      auroc_mean, auroc_ci95_low, auroc_ci95_high, auprc_mean, ...
    """

    long = summarize_metrics(metrics_per_split)
    cells = {}
    for _, r in long.iterrows():
        m = str(r["metric"])
        cells[f"{m}_mean"] = r["mean"]
        cells[f"{m}_ci95_low"] = r["ci95_low"]
        cells[f"{m}_ci95_high"] = r["ci95_high"]
        cells[f"{m}_n"] = r["n"]
    return pd.DataFrame([cells])
