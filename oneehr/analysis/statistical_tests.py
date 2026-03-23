"""Statistical tests for pairwise model comparison: DeLong, McNemar, bootstrap CI."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _delong_roc_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[float, float]:
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


def _mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[float, float]:
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


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a metric function.

    Parameters
    ----------
    metric_fn : callable(y_true, y_pred) -> float
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            continue

    if not scores:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(np.percentile(scores, 100 * alpha)),
        "ci_upper": float(np.percentile(scores, 100 * (1 - alpha))),
        "std": float(np.std(scores)),
    }


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_kind: str = "binary",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap CI for all standard metrics."""
    from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

    results = {}
    if task_kind == "binary":
        metric_fns = {
            "auroc": lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) >= 2 else float("nan"),
            "auprc": lambda yt, yp: average_precision_score(yt, yp),
            "brier": lambda yt, yp: float(np.mean((yp - yt) ** 2)),
            "mcc": lambda yt, yp: matthews_corrcoef(yt, (yp >= 0.5).astype(int)),
        }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        metric_fns = {
            "mae": lambda yt, yp: mean_absolute_error(yt, yp),
            "rmse": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        }

    for name, fn in metric_fns.items():
        results[name] = bootstrap_metric_ci(
            y_true,
            y_pred,
            fn,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return results


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------


def _bonferroni_correction(p_values: list[float]) -> list[float]:
    """Bonferroni correction: multiply p-values by number of comparisons."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def _bh_fdr_correction(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adj = min(p * n / rank, prev)
        corrected[orig_idx] = min(adj, 1.0)
        prev = adj
    return corrected


def compute_statistical_tests(
    *,
    preds: pd.DataFrame,
    n_bootstrap: int = 1000,
    correction: str = "bh",
) -> dict:
    """Run pairwise DeLong and McNemar tests between all systems.

    Parameters
    ----------
    correction : "bonferroni", "bh" (Benjamini-Hochberg FDR), or "none"
    n_bootstrap : number of bootstrap resamples for CI computation
    """
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
    delong_pvals = []
    mcnemar_pvals = []

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
        delong_pvals.append(delong_p)
        mcnemar_pvals.append(mcnemar_p)

        pairwise.append(
            {
                "system_a": a,
                "system_b": b,
                "n": int(n),
                "delong": {"z_stat": z, "p_value": delong_p},
                "mcnemar": {"chi2": chi2, "p_value": mcnemar_p},
            }
        )

    # Apply multiple comparison correction
    if correction == "bonferroni":
        delong_adj = _bonferroni_correction(delong_pvals)
        mcnemar_adj = _bonferroni_correction(mcnemar_pvals)
    elif correction == "bh":
        delong_adj = _bh_fdr_correction(delong_pvals)
        mcnemar_adj = _bh_fdr_correction(mcnemar_pvals)
    else:
        delong_adj = delong_pvals
        mcnemar_adj = mcnemar_pvals

    for i, pw in enumerate(pairwise):
        pw["delong"]["p_adjusted"] = delong_adj[i]
        pw["mcnemar"]["p_adjusted"] = mcnemar_adj[i]

    # Bootstrap CI for each system
    bootstrap_cis = {}
    for s in systems:
        yt, yp = system_data[s]
        bootstrap_cis[s] = bootstrap_all_metrics(yt, yp, n_bootstrap=n_bootstrap)

    return {
        "module": "statistical_tests",
        "n_common_patients": len(common_sorted),
        "correction_method": correction,
        "pairwise": pairwise,
        "bootstrap_ci": bootstrap_cis,
    }
