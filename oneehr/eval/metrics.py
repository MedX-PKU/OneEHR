from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    metrics: dict[str, float]


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> MetricResult:
    """Standard binary classification metrics.

    Computes AUROC, AUPRC, and threshold-based metrics.
    """

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
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

    # Probabilistic metric.
    out["logloss"] = float(log_loss(y_true, y_score, labels=[0, 1]))
    return MetricResult(metrics=out)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    out: dict[str, float] = {}
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["mse"] = float(mean_squared_error(y_true, y_pred, squared=True))
    out["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
    out["r2"] = float(r2_score(y_true, y_pred))
    return MetricResult(metrics=out)
