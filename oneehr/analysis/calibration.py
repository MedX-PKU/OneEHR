"""Post-hoc calibration methods: temperature scaling, Platt, isotonic."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _temperature_scale(val_probs: np.ndarray, val_labels: np.ndarray) -> tuple[float, callable]:
    """Fit temperature scaling on validation data."""
    from scipy.optimize import minimize

    def nll(T):
        T = max(T[0], 1e-6)
        logits = np.log(np.clip(val_probs, 1e-7, 1 - 1e-7))
        scaled = logits / T
        probs = 1 / (1 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(val_labels * np.log(probs) + (1 - val_labels) * np.log(1 - probs))

    result = minimize(nll, x0=[1.0], bounds=[(0.01, 20.0)], method="L-BFGS-B")
    T_opt = float(result.x[0])

    def transform(probs):
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7))
        scaled = logits / T_opt
        return 1 / (1 + np.exp(-scaled))

    return T_opt, transform


def _platt_scale(val_probs: np.ndarray, val_labels: np.ndarray) -> callable:
    """Fit Platt scaling (logistic regression on logits)."""
    from sklearn.linear_model import LogisticRegression

    logits = np.log(np.clip(val_probs, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logits, val_labels.astype(int))

    def transform(probs):
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        return lr.predict_proba(logits)[:, 1]

    return transform


def _isotonic_scale(val_probs: np.ndarray, val_labels: np.ndarray) -> callable:
    """Fit isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(val_probs, val_labels)
    return iso.predict


def _ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score > lo) & (y_score <= hi)
        if lo == 0.0:
            mask |= y_score == 0.0
        count = mask.sum()
        if count == 0:
            continue
        ece += (count / n) * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def _brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


def compute_calibration(
    *,
    preds: pd.DataFrame,
    split_info: dict,
) -> tuple[dict, pd.DataFrame]:
    """Fit calibration methods on val predictions, evaluate on test.

    Parameters
    ----------
    preds : predictions.parquet
    split_info : split.json with val/test patient lists

    Returns
    -------
    results : dict of per-system, per-method calibration metrics
    calibrated_preds : DataFrame with recalibrated probability columns
    """
    val_patients = set(str(p) for p in split_info.get("val", []))
    test_patients = set(str(p) for p in split_info.get("test", []))

    preds = preds.copy()
    preds["patient_id"] = preds["patient_id"].astype(str)

    methods = {
        "temperature_scaling": _temperature_scale,
        "platt_scaling": _platt_scale,
        "isotonic_regression": _isotonic_scale,
    }

    systems = []
    all_calibrated = []

    for system_name in preds["system"].unique():
        sdf = preds[preds["system"] == system_name].copy()
        val_mask = sdf["patient_id"].isin(val_patients)
        test_mask = sdf["patient_id"].isin(test_patients)

        val_df = sdf[val_mask]
        test_df = sdf[test_mask]

        y_val_true = val_df["y_true"].to_numpy(dtype=float)
        y_val_pred = val_df["y_pred"].to_numpy(dtype=float)
        y_test_true = test_df["y_true"].to_numpy(dtype=float)
        y_test_pred = test_df["y_pred"].to_numpy(dtype=float)

        if y_val_true.size == 0 or y_test_true.size == 0:
            continue

        # Pre-calibration metrics on test
        pre_ece = _ece(y_test_true, y_test_pred)
        pre_brier = _brier(y_test_true, y_test_pred)

        method_results = {}
        cal_row = test_df[["patient_id", "system", "y_true", "y_pred"]].copy()

        for method_name, fit_fn in methods.items():
            try:
                if method_name == "temperature_scaling":
                    T_opt, transform = fit_fn(y_val_pred, y_val_true)
                    extra = {"temperature": T_opt}
                else:
                    transform = fit_fn(y_val_pred, y_val_true)
                    extra = {}

                y_cal = np.clip(transform(y_test_pred), 0.0, 1.0)
                post_ece = _ece(y_test_true, y_cal)
                post_brier = _brier(y_test_true, y_cal)

                method_results[method_name] = {
                    "pre_ece": pre_ece,
                    "pre_brier": pre_brier,
                    "post_ece": post_ece,
                    "post_brier": post_brier,
                    **extra,
                }
                cal_row[f"y_pred_{method_name}"] = y_cal
            except Exception as e:
                method_results[method_name] = {"error": str(e)}

        systems.append(
            {
                "name": system_name,
                "n_val": int(y_val_true.size),
                "n_test": int(y_test_true.size),
                "methods": method_results,
            }
        )
        all_calibrated.append(cal_row)

    calibrated_df = pd.concat(all_calibrated, ignore_index=True) if all_calibrated else pd.DataFrame()

    return {
        "module": "calibration",
        "systems": systems,
    }, calibrated_df
