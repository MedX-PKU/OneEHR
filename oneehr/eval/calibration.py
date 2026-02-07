from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    def logits_to_proba(self, logits: np.ndarray) -> np.ndarray:
        t = float(self.temperature)
        if not np.isfinite(t) or t <= 0:
            raise ValueError(f"Invalid temperature={t}")
        return sigmoid(np.asarray(logits, dtype=float) / t)


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float

    def logits_to_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        return sigmoid(self.a * logits + self.b)


def fit_temperature_scaling(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    max_iter: int = 200,
    lr: float = 0.1,
    min_temperature: float = 1e-3,
) -> TemperatureCalibrator:
    """Fit temperature scaling on a calibration set.

    Uses simple gradient descent on log-temperature for numerical stability.
    """

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if y_true.shape != logits.shape:
        raise ValueError("y_true and logits must have same shape")

    if np.unique(y_true).size < 2:
        # Degenerate calibration set; keep identity transform.
        return TemperatureCalibrator(temperature=1.0)

    s = 0.0  # log(T)
    for _ in range(int(max_iter)):
        t = float(max(np.exp(s), min_temperature))
        z = logits / t
        p = sigmoid(z)
        # dL/dz = p - y
        # z = logits / t, dt/ds = t, dz/ds = d(logits/t)/ds = -logits/t
        grad = np.mean((p - y_true) * (-logits / t))
        if not np.isfinite(grad):
            break
        s -= float(lr) * float(grad)

    t = float(max(np.exp(s), min_temperature))
    return TemperatureCalibrator(temperature=t)


def fit_platt_scaling(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    max_iter: int = 400,
    lr: float = 0.1,
    l2: float = 0.0,
) -> PlattCalibrator:
    """Fit Platt scaling p=sigmoid(a*logit+b) by gradient descent."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if y_true.shape != logits.shape:
        raise ValueError("y_true and logits must have same shape")

    if np.unique(y_true).size < 2:
        return PlattCalibrator(a=1.0, b=0.0)

    a = 1.0
    b = 0.0
    for _ in range(int(max_iter)):
        z = a * logits + b
        p = sigmoid(z)
        # dL/dz = p - y
        da = np.mean((p - y_true) * logits) + float(l2) * a
        db = np.mean(p - y_true)
        if not (np.isfinite(da) and np.isfinite(db)):
            break
        a -= float(lr) * float(da)
        b -= float(lr) * float(db)

    return PlattCalibrator(a=float(a), b=float(b))


def calibrate_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    method: str,
    temperature_kwargs: dict[str, object] | None = None,
    platt_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Convenience API for calibrating when you only have probabilities.

    Internally converts probs -> logits, then delegates to `calibrate_from_logits`.
    """

    probs = np.asarray(probs, dtype=float).reshape(-1)
    logits = _logit(probs)
    return calibrate_from_logits(
        y_true,
        logits,
        method=method,
        temperature_kwargs=temperature_kwargs,
        platt_kwargs=platt_kwargs,
    )


def calibrate_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    method: str,
    temperature_kwargs: dict[str, object] | None = None,
    platt_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if method == "temperature":
        kw = {} if temperature_kwargs is None else dict(temperature_kwargs)
        cal = fit_temperature_scaling(y_true, logits, **kw)
        p_cal = cal.logits_to_proba(logits)
        return p_cal, {"temperature": float(cal.temperature)}
    if method == "platt":
        kw = {} if platt_kwargs is None else dict(platt_kwargs)
        cal = fit_platt_scaling(y_true, logits, **kw)
        p_cal = cal.logits_to_proba(logits)
        return p_cal, {"a": float(cal.a), "b": float(cal.b)}
    raise ValueError(f"Unsupported calibration method={method!r}")


def select_threshold_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    if y_true.shape != probs.shape:
        raise ValueError("y_true and probs must have same shape")
    if np.unique(y_true).size < 2:
        return 0.5

    order = np.argsort(-probs)
    y = y_true[order]
    p = probs[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    fn = float(np.sum(y)) - tp

    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    best = int(np.argmax(f1))
    return float(p[best])


def binary_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    eps = np.finfo(float).eps
    probs = np.clip(probs, eps, 1.0 - eps)
    loss = -(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
    return float(np.mean(loss))


def binary_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    return float(np.mean((probs - y_true) ** 2))
