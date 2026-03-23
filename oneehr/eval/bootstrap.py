from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oneehr.config.schema import TaskConfig
from oneehr.eval.metrics import binary_metrics, regression_metrics


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    values: list[float]


def bootstrap_metric(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: TaskConfig,
    metric: str,
    n: int = 200,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    if n <= 1:
        raise ValueError("bootstrap n must be > 1")

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have same length")

    mvals: list[float] = []
    idx = np.arange(y_true.shape[0])
    for _ in range(int(n)):
        samp = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[samp]
        yp = y_pred[samp]
        if task.kind == "binary":
            mets = binary_metrics(yt.astype(float), yp.astype(float)).metrics
        else:
            mets = regression_metrics(yt.astype(float), yp.astype(float)).metrics
        if metric not in mets:
            raise ValueError(f"Unknown metric {metric!r} for task.kind={task.kind!r}")
        mvals.append(float(mets[metric]))

    mean = float(np.mean(mvals))
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.quantile(mvals, alpha))
    ci_high = float(np.quantile(mvals, 1.0 - alpha))
    return BootstrapResult(
        metric=metric,
        n=int(n),
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        values=mvals,
    )
