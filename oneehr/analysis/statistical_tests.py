"""Statistical tests for pairwise model comparison: DeLong and McNemar."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _delong_roc_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> tuple[float, float]:
    """DeLong test for comparing two AUROCs.

    Returns (z_stat, p_value).
    Based on the DeLong et al. (1988) method.
    """
    from scipy import stats

    pos = y_true == 1
    neg = y_true == 0
    n1 = pos.sum()
    n0 = neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")

    pred_a_pos = y_pred_a[pos]
    pred_a_neg = y_pred_a[neg]
    pred_b_pos = y_pred_b[pos]
    pred_b_neg = y_pred_b[neg]

    # Structural components (placement values)
    V_a10 = np.array([(pred_a_neg < p).mean() + 0.5 * (pred_a_neg == p).mean() for p in pred_a_pos])
    V_a01 = np.array([(pred_a_pos > n).mean() + 0.5 * (pred_a_pos == n).mean() for n in pred_a_neg])
    V_b10 = np.array([(pred_b_neg < p).mean() + 0.5 * (pred_b_neg == p).mean() for p in pred_b_pos])
    V_b01 = np.array([(pred_b_pos > n).mean() + 0.5 * (pred_b_pos == n).mean() for n in pred_b_neg])

    auc_a = V_a10.mean()
    auc_b = V_b10.mean()

    # Covariance
    S10 = np.cov(np.stack([V_a10, V_b10])) if n1 > 1 else np.zeros((2, 2))
    S01 = np.cov(np.stack([V_a01, V_b01])) if n0 > 1 else np.zeros((2, 2))

    S = S10 / n1 + S01 / n0

    # Variance of difference
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p)


def _mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> tuple[float, float]:
    """McNemar test comparing error rates of two classifiers.

    Returns (chi2, p_value).
    """
    from scipy import stats

    y_hat_a = (y_pred_a >= 0.5).astype(int)
    y_hat_b = (y_pred_b >= 0.5).astype(int)
    correct_a = (y_hat_a == y_true).astype(int)
    correct_b = (y_hat_b == y_true).astype(int)

    # b = A wrong, B right; c = A right, B wrong
    b = ((correct_a == 0) & (correct_b == 1)).sum()
    c = ((correct_a == 1) & (correct_b == 0)).sum()

    if b + c == 0:
        return 0.0, 1.0

    # Continuity-corrected McNemar
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), float(p)


def compute_statistical_tests(*, preds: pd.DataFrame) -> dict:
    """Run pairwise DeLong and McNemar tests between all systems."""
    systems = sorted(preds["system"].unique())
    if len(systems) < 2:
        return {"module": "statistical_tests", "pairwise": [], "note": "need >= 2 systems"}

    # Build aligned arrays per system
    system_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_patients = set()
    for s in systems:
        sdf = preds[preds["system"] == s].copy()
        sdf["patient_id"] = sdf["patient_id"].astype(str)
        all_patients.update(sdf["patient_id"].tolist())

    # Only use patients present in ALL systems
    common_patients = None
    for s in systems:
        sdf = preds[preds["system"] == s].copy()
        sdf["patient_id"] = sdf["patient_id"].astype(str)
        pts = set(sdf["patient_id"].tolist())
        common_patients = pts if common_patients is None else common_patients & pts

    if not common_patients:
        return {"module": "statistical_tests", "pairwise": [], "note": "no common patients"}

    common_sorted = sorted(common_patients)
    for s in systems:
        sdf = preds[preds["system"] == s].copy()
        sdf["patient_id"] = sdf["patient_id"].astype(str)
        sdf = sdf.set_index("patient_id").loc[common_sorted]
        y_true = sdf["y_true"].to_numpy(dtype=float)
        y_pred = sdf["y_pred"].to_numpy(dtype=float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        system_data[s] = (y_true[finite], y_pred[finite])

    pairwise = []
    for a, b in combinations(systems, 2):
        y_true_a, y_pred_a = system_data[a]
        y_true_b, y_pred_b = system_data[b]
        # They should be aligned to same patients, so y_true should match
        n = min(len(y_true_a), len(y_true_b))
        yt = y_true_a[:n]
        ypa = y_pred_a[:n]
        ypb = y_pred_b[:n]

        z, delong_p = _delong_roc_test(yt, ypa, ypb)
        chi2, mcnemar_p = _mcnemar_test(yt, ypa, ypb)

        pairwise.append({
            "system_a": a,
            "system_b": b,
            "n": int(n),
            "delong": {"z_stat": z, "p_value": delong_p},
            "mcnemar": {"chi2": chi2, "p_value": mcnemar_p},
        })

    return {"module": "statistical_tests", "n_common_patients": len(common_sorted), "pairwise": pairwise}
