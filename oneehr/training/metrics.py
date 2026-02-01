from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    metrics: dict[str, float]


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> MetricResult:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out: dict[str, float] = {}
    out["auc"] = float(roc_auc_score(y_true, y_score))
    out["ap"] = float(average_precision_score(y_true, y_score))
    return MetricResult(metrics=out)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    out: dict[str, float] = {}
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
    return MetricResult(metrics=out)

