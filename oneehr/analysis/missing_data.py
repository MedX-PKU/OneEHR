"""Missing data quality report from preprocessed binned data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_missing_data(*, binned: pd.DataFrame) -> dict:
    """Analyze missingness patterns in binned feature data.

    Parameters
    ----------
    binned : binned.parquet with feature columns (num__*, cat__*)

    Returns
    -------
    dict with per-feature missingness rates, correlation matrix summary,
    and top correlated missing pairs.
    """
    feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    if not feat_cols:
        return {"module": "missing_data", "features": [], "note": "no feature columns"}

    n_rows = len(binned)
    if n_rows == 0:
        return {"module": "missing_data", "features": [], "note": "empty data"}

    # Per-feature missingness
    miss_rates = {}
    for col in feat_cols:
        n_miss = int(binned[col].isna().sum())
        miss_rates[col] = {
            "n_missing": n_miss,
            "n_total": n_rows,
            "rate": round(n_miss / n_rows, 6),
        }

    # Filter to features that actually have missing values
    cols_with_missing = [c for c in feat_cols if miss_rates[c]["n_missing"] > 0]

    # Missingness correlation matrix (indicator matrix)
    corr_pairs = []
    if len(cols_with_missing) >= 2:
        miss_indicators = binned[cols_with_missing].isna().astype(float)
        # Limit to top 50 features by missingness rate to keep computation bounded
        top_cols = sorted(cols_with_missing, key=lambda c: miss_rates[c]["rate"], reverse=True)[:50]
        if len(top_cols) >= 2:
            corr = miss_indicators[top_cols].corr()
            # Extract top correlated pairs
            seen = set()
            for i, c1 in enumerate(top_cols):
                for j, c2 in enumerate(top_cols):
                    if i >= j:
                        continue
                    key = (c1, c2)
                    if key in seen:
                        continue
                    seen.add(key)
                    r = corr.loc[c1, c2]
                    if np.isfinite(r) and abs(r) > 0.1:
                        corr_pairs.append({
                            "feature_a": c1,
                            "feature_b": c2,
                            "correlation": round(float(r), 4),
                        })

    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Summary stats
    rates = [v["rate"] for v in miss_rates.values()]
    summary = {
        "n_features": len(feat_cols),
        "n_features_with_missing": len(cols_with_missing),
        "mean_missing_rate": round(float(np.mean(rates)), 6) if rates else 0.0,
        "max_missing_rate": round(float(np.max(rates)), 6) if rates else 0.0,
    }

    return {
        "module": "missing_data",
        "summary": summary,
        "features": miss_rates,
        "top_correlated_missing_pairs": corr_pairs[:20],
    }
