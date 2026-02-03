from __future__ import annotations

import numpy as np
import pandas as pd


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


def summarize_metrics(
    metrics_per_split: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create a long table with mean/std/bootstrap 95% CI over splits."""

    if "split" not in metrics_per_split.columns:
        raise ValueError("metrics_per_split must contain 'split'")

    group_cols = group_cols or []
    ignore_cols = {"split", "hpo_best", "reason", "model"}
    ignore_cols.update(group_cols)

    metric_cols = [
        c
        for c in metrics_per_split.columns
        if c not in ignore_cols and not c.startswith("skipped")
    ]

    def _summarize_block(block: pd.DataFrame) -> list[dict]:
        rows = []
        for col in metric_cols:
            s = pd.to_numeric(block[col], errors="coerce")
            if s.notna().sum() == 0:
                continue
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
        return rows

    if group_cols:
        all_rows: list[dict] = []
        for keys, block in metrics_per_split.groupby(group_cols):
            keys = (keys,) if not isinstance(keys, tuple) else keys
            block_rows = _summarize_block(block)
            for row in block_rows:
                for col, key in zip(group_cols, keys, strict=False):
                    row[col] = key
                all_rows.append(row)
        if not all_rows:
            cols = [*group_cols, "metric", "mean", "std", "ci95_low", "ci95_high", "n"]
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(all_rows).sort_values([*group_cols, "metric"])

    rows = _summarize_block(metrics_per_split)
    if not rows:
        return pd.DataFrame(columns=["metric", "mean", "std", "ci95_low", "ci95_high", "n"])
    return pd.DataFrame(rows).sort_values("metric")


def to_paper_wide_table(
    metrics_per_split: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a wide table with mean/CI columns."""

    group_cols = group_cols or []
    long = summarize_metrics(metrics_per_split, group_cols=group_cols)
    if long.empty:
        return pd.DataFrame()

    if not group_cols:
        cells = {}
        for _, r in long.iterrows():
            m = str(r["metric"])
            cells[f"{m}_mean"] = r["mean"]
            cells[f"{m}_ci95_low"] = r["ci95_low"]
            cells[f"{m}_ci95_high"] = r["ci95_high"]
            cells[f"{m}_n"] = r["n"]
        return pd.DataFrame([cells])

    rows = []
    for keys, block in long.groupby(group_cols):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        cells = {col: key for col, key in zip(group_cols, keys, strict=False)}
        for _, r in block.iterrows():
            m = str(r["metric"])
            cells[f"{m}_mean"] = r["mean"]
            cells[f"{m}_ci95_low"] = r["ci95_low"]
            cells[f"{m}_ci95_high"] = r["ci95_high"]
            cells[f"{m}_n"] = r["n"]
        rows.append(cells)
    return pd.DataFrame(rows)
