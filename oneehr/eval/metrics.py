from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _ece_binary(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width binning."""
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
        avg_conf = y_score[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)
    return ece


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> MetricResult:
    """Standard binary classification metrics.

    Computes AUROC, AUPRC, threshold-based metrics, and clinical metrics.
    """

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    out: dict[str, float] = {}
    if np.unique(y_true).size < 2:
        out["auroc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true, y_score))

    # AUPRC is defined for single-class y_true but can be uninformative.
    out["auprc"] = float(average_precision_score(y_true, y_score))

    # Threshold-based metrics (default 0.5).
    y_hat = (y_score >= 0.5).astype(int)
    out["accuracy"] = float(accuracy_score(y_true, y_hat))
    out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))

    # MCC (Matthews Correlation Coefficient)
    out["mcc"] = float(matthews_corrcoef(y_true, y_hat))

    # Sensitivity / Specificity at threshold 0.5
    tp = ((y_hat == 1) & (y_true == 1)).sum()
    fn = ((y_hat == 0) & (y_true == 1)).sum()
    fp = ((y_hat == 1) & (y_true == 0)).sum()
    tn = ((y_hat == 0) & (y_true == 0)).sum()
    out["sensitivity"] = float(tp / max(tp + fn, 1))
    out["specificity"] = float(tn / max(tn + fp, 1))

    # Youden index (optimal threshold maximizing sensitivity + specificity - 1)
    if np.unique(y_true).size >= 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        out["youden_index"] = float(youden[best_idx])
        out["youden_threshold"] = float(thresholds[best_idx])
    else:
        out["youden_index"] = float("nan")
        out["youden_threshold"] = float("nan")

    # Probabilistic metrics.
    out["logloss"] = float(log_loss(y_true, y_score, labels=[0, 1]))

    # Calibration metrics.
    out["brier"] = float(np.mean((y_score - y_true) ** 2))
    out["ece"] = float(_ece_binary(y_true, y_score))

    return MetricResult(metrics=out)


def sensitivity_specificity_at_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute sensitivity and specificity at given thresholds."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    for t in thresholds:
        y_hat = (y_score >= t).astype(int)
        tp = ((y_hat == 1) & (y_true == 1)).sum()
        fn = ((y_hat == 0) & (y_true == 1)).sum()
        fp = ((y_hat == 1) & (y_true == 0)).sum()
        tn = ((y_hat == 0) & (y_true == 0)).sum()
        results[f"t={t:.2f}"] = {
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "ppv": float(tp / max(tp + fp, 1)),
            "npv": float(tn / max(tn + fn, 1)),
        }
    return results


def net_benefit(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Decision Curve Analysis: compute net benefit at each threshold.

    Net benefit = TP/N - FP/N * (pt / (1 - pt))
    where pt is the threshold probability.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    n = len(y_true)
    net_benefits = []
    treat_all = []
    for pt in thresholds:
        y_hat = (y_score >= pt).astype(int)
        tp = ((y_hat == 1) & (y_true == 1)).sum()
        fp = ((y_hat == 1) & (y_true == 0)).sum()
        nb = tp / n - fp / n * (pt / (1 - pt))
        net_benefits.append(float(nb))
        # Treat-all reference
        prevalence = y_true.mean()
        ta = prevalence - (1 - prevalence) * (pt / (1 - pt))
        treat_all.append(float(ta))
    return {
        "thresholds": thresholds.tolist(),
        "net_benefit_model": net_benefits,
        "net_benefit_treat_all": treat_all,
    }


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------

def multiclass_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    num_classes: int,
) -> MetricResult:
    """Multiclass classification metrics.

    Parameters
    ----------
    y_true : (N,) integer class labels
    y_score : (N, C) class probabilities
    num_classes : number of classes
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    out: dict[str, float] = {}

    y_pred = y_score.argmax(axis=1) if y_score.ndim == 2 else y_score.astype(int)

    out["accuracy"] = float(accuracy_score(y_true, y_pred))

    for avg in ("macro", "micro", "weighted"):
        out[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        out[f"precision_{avg}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        out[f"recall_{avg}"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    # Per-class and macro AUROC (one-vs-rest)
    if y_score.ndim == 2 and len(np.unique(y_true)) >= 2:
        try:
            out["auroc_macro"] = float(roc_auc_score(
                y_true, y_score, multi_class="ovr", average="macro",
            ))
            out["auroc_weighted"] = float(roc_auc_score(
                y_true, y_score, multi_class="ovr", average="weighted",
            ))
        except ValueError:
            out["auroc_macro"] = float("nan")
            out["auroc_weighted"] = float("nan")
    else:
        out["auroc_macro"] = float("nan")
        out["auroc_weighted"] = float("nan")

    return MetricResult(metrics=out)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    out: dict[str, float] = {}
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    # scikit-learn removed the `squared` kwarg from mean_squared_error in 1.8.
    mse = float(mean_squared_error(y_true, y_pred))
    out["mse"] = mse
    out["rmse"] = float(np.sqrt(mse))
    out["r2"] = float(r2_score(y_true, y_pred))
    return MetricResult(metrics=out)


# ---------------------------------------------------------------------------
# Multi-label classification
# ---------------------------------------------------------------------------

def multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> MetricResult:
    """Multi-label classification metrics.

    Parameters
    ----------
    y_true : (N, L) binary label matrix
    y_score : (N, L) predicted probabilities
    threshold : float
        Binarization threshold for y_score.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_score >= threshold).astype(int)
    out: dict[str, float] = {}

    # Subset accuracy (exact match)
    out["subset_accuracy"] = float(accuracy_score(y_true, y_pred))
    out["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    for avg in ("macro", "micro", "weighted", "samples"):
        out[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        out[f"precision_{avg}"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        out[f"recall_{avg}"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    # AUROC
    try:
        out["auroc_macro"] = float(roc_auc_score(y_true, y_score, average="macro"))
        out["auroc_micro"] = float(roc_auc_score(y_true, y_score, average="micro"))
    except ValueError:
        out["auroc_macro"] = float("nan")
        out["auroc_micro"] = float("nan")

    return MetricResult(metrics=out)
