from __future__ import annotations

import numpy as np

from oneehr.eval.calibration import (
    binary_log_loss,
    calibrate_from_logits,
    select_threshold_f1,
)


def test_temperature_calibration_improves_logloss_on_miscalibrated_logits() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    p_true = 1.0 / (1.0 + np.exp(-x))
    y = (rng.random(n) < p_true).astype(float)

    # Overconfident model: logits scaled up.
    logits = 3.0 * x
    p_raw = 1.0 / (1.0 + np.exp(-logits))

    p_cal, params = calibrate_from_logits(y, logits, method="temperature")
    assert "temperature" in params

    ll_raw = binary_log_loss(y, p_raw)
    ll_cal = binary_log_loss(y, p_cal)
    assert ll_cal < ll_raw


def test_select_threshold_f1_returns_valid_probability() -> None:
    y = np.array([0, 1, 1, 0, 1], dtype=float)
    p = np.array([0.1, 0.6, 0.4, 0.2, 0.9], dtype=float)
    thr = select_threshold_f1(y, p)
    assert 0.0 <= thr <= 1.0

