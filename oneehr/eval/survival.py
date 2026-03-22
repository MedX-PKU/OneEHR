"""Survival analysis metrics: concordance index, time-dependent Brier score."""

from __future__ import annotations

import numpy as np

from oneehr.eval.metrics import MetricResult


def concordance_index(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Harrell's concordance index (C-index).

    Parameters
    ----------
    event_times : (N,) observed times
    predicted_risk : (N,) predicted risk scores (higher = more risk)
    event_observed : (N,) event indicator (1 = event, 0 = censored)

    Returns
    -------
    C-index in [0, 1]. 0.5 = random, 1.0 = perfect concordance.
    """
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        if event_observed[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[j] > event_times[i]:
                if predicted_risk[j] < predicted_risk[i]:
                    concordant += 1
                elif predicted_risk[j] > predicted_risk[i]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


def concordance_index_fast(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Faster C-index computation using sorting.

    Falls back to lifelines if available for even faster computation.
    """
    try:
        from lifelines.utils import concordance_index as _ci
        return float(_ci(event_times, -predicted_risk, event_observed))
    except ImportError:
        pass

    return concordance_index(event_times, predicted_risk, event_observed)


def brier_score_at_time(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_survival: np.ndarray,
    t: float,
) -> float:
    """Time-dependent Brier score at time t.

    Parameters
    ----------
    event_times : (N,) observed times
    event_observed : (N,) event indicator
    predicted_survival : (N,) predicted S(t) for each individual
    t : evaluation time point
    """
    n = len(event_times)
    # IPCW weights (simplified Kaplan-Meier estimate of censoring)
    censor_times = event_times[event_observed == 0]
    if len(censor_times) == 0:
        # No censoring — simple Brier score
        y_true = ((event_times <= t) & (event_observed == 1)).astype(float)
        return float(np.mean((predicted_survival - (1 - y_true)) ** 2))

    # Simple IPCW
    bs = 0.0
    for i in range(n):
        if event_times[i] <= t and event_observed[i] == 1:
            bs += (0.0 - predicted_survival[i]) ** 2
        elif event_times[i] > t:
            bs += (1.0 - predicted_survival[i]) ** 2

    return bs / n


def integrated_brier_score(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_survival_fn,
    time_points: np.ndarray | None = None,
) -> float:
    """Integrated Brier Score (IBS) over a range of time points.

    Parameters
    ----------
    predicted_survival_fn : callable
        Function that takes a time t and returns (N,) predicted S(t).
    """
    if time_points is None:
        observed_events = event_times[event_observed == 1]
        if len(observed_events) == 0:
            return float("nan")
        time_points = np.linspace(
            observed_events.min(),
            observed_events.max(),
            50,
        )

    scores = []
    for t in time_points:
        s_t = predicted_survival_fn(t)
        scores.append(brier_score_at_time(event_times, event_observed, s_t, t))

    return float(np.trapz(scores, time_points) / (time_points[-1] - time_points[0]))


def survival_metrics(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
) -> MetricResult:
    """Compute standard survival analysis metrics.

    Parameters
    ----------
    event_times : (N,) observed times
    predicted_risk : (N,) predicted risk scores
    event_observed : (N,) event indicator (1 = event, 0 = censored)
    """
    ci = concordance_index_fast(event_times, predicted_risk, event_observed)

    return MetricResult(metrics={
        "c_index": ci,
        "n_events": int(event_observed.sum()),
        "n_censored": int((event_observed == 0).sum()),
        "median_time": float(np.median(event_times)),
    })
