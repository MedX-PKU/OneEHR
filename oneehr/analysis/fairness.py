"""Fairness / bias analysis across sensitive attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oneehr.eval.metrics import binary_metrics


_SENSITIVE_PATTERNS = ("age", "sex", "gender", "race", "ethnicity")


def _detect_sensitive_columns(static: pd.DataFrame) -> list[str]:
    """Auto-detect columns whose names match common sensitive attribute patterns."""
    out = []
    for col in static.columns:
        low = str(col).lower().replace("num__", "").replace("cat__", "")
        if any(pat in low for pat in _SENSITIVE_PATTERNS):
            out.append(col)
    return out


def _demographic_parity_diff(groups: dict[str, np.ndarray]) -> float:
    """Max absolute difference of mean predicted probability across groups."""
    means = [y.mean() for y in groups.values() if len(y) > 0]
    if len(means) < 2:
        return 0.0
    return float(max(means) - min(means))


def _equalized_odds_diff(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    """Max absolute difference of TPR and FPR across groups."""
    tprs, fprs = [], []
    for y_true, y_pred in groups.values():
        y_hat = (y_pred >= 0.5).astype(int)
        pos = y_true == 1
        neg = y_true == 0
        tpr = y_hat[pos].mean() if pos.sum() > 0 else 0.0
        fpr = y_hat[neg].mean() if neg.sum() > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    if len(tprs) < 2:
        return 0.0
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def _smd_predictions(groups: dict[str, np.ndarray]) -> float:
    """Standardized mean difference of predictions between groups."""
    means = []
    stds = []
    for y in groups.values():
        if len(y) == 0:
            continue
        means.append(y.mean())
        stds.append(y.std())
    if len(means) < 2:
        return 0.0
    pooled_std = np.sqrt(np.mean(np.array(stds) ** 2))
    if pooled_std < 1e-12:
        return 0.0
    return float(abs(means[0] - means[1]) / pooled_std) if len(means) == 2 else float(
        (max(means) - min(means)) / pooled_std
    )


def compute_fairness(
    *,
    preds: pd.DataFrame,
    static: pd.DataFrame,
    sensitive_columns: list[str] | None = None,
) -> dict:
    """Compute per-system fairness metrics across sensitive attributes.

    Parameters
    ----------
    preds : predictions.parquet with columns: patient_id, system, y_true, y_pred
    static : static.parquet with patient_id index or column
    sensitive_columns : override auto-detected sensitive columns
    """
    if sensitive_columns is None:
        sensitive_columns = _detect_sensitive_columns(static)
    if not sensitive_columns:
        return {"sensitive_columns": [], "systems": []}

    # Ensure static has patient_id as column
    if "patient_id" not in static.columns:
        static = static.reset_index()
    static = static.copy()
    static["patient_id"] = static["patient_id"].astype(str)

    preds = preds.copy()
    preds["patient_id"] = preds["patient_id"].astype(str)

    results = {"sensitive_columns": sensitive_columns, "systems": []}

    for system_name in preds["system"].unique():
        sdf = preds[preds["system"] == system_name].merge(
            static[["patient_id", *sensitive_columns]], on="patient_id", how="left"
        )
        y_true = sdf["y_true"].to_numpy(dtype=float)
        y_pred = sdf["y_pred"].to_numpy(dtype=float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        sdf = sdf[finite]
        y_true = y_true[finite]
        y_pred = y_pred[finite]

        attr_results = {}
        for col in sensitive_columns:
            if col not in sdf.columns:
                continue
            vals = sdf[col].fillna("__missing__").astype(str)
            group_names = sorted(vals.unique())

            group_metrics = {}
            pred_groups: dict[str, np.ndarray] = {}
            truepred_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}

            for g in group_names:
                mask = (vals == g).to_numpy()
                yt = y_true[mask]
                yp = y_pred[mask]
                if yt.size == 0:
                    continue
                pred_groups[g] = yp
                truepred_groups[g] = (yt, yp)
                m = binary_metrics(yt, yp).metrics
                group_metrics[g] = {"n": int(yt.size), **m}

            attr_results[col] = {
                "groups": group_metrics,
                "demographic_parity_diff": _demographic_parity_diff(pred_groups),
                "equalized_odds_diff": _equalized_odds_diff(truepred_groups),
                "smd_predictions": _smd_predictions(pred_groups),
            }

        results["systems"].append({
            "name": system_name,
            "n": int(y_true.size),
            "attributes": attr_results,
        })

    return results
