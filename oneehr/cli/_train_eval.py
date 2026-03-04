"""Calibration, metrics, and evaluation helpers for train pipeline."""
from __future__ import annotations

import sys

import numpy as np

from oneehr.eval.calibration import sigmoid


def maybe_calibrate_and_threshold(
    *,
    cfg0,
    y_val_true: np.ndarray,
    y_val_score: np.ndarray,
    y_val_logits: np.ndarray | None,
    y_test_score: np.ndarray,
    y_test_logits: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply optional calibration using val split, and compute thresholds."""

    from oneehr.eval.calibration import (
        binary_brier,
        binary_log_loss,
        calibrate_from_logits,
        calibrate_from_probs,
        select_threshold_f1,
    )

    extra: dict[str, float] = {}
    if cfg0.task.kind != "binary" or not cfg0.calibration.enabled:
        return y_test_score, extra

    if cfg0.calibration.source != "val":
        raise SystemExit("calibration.source currently supports 'val' only")
    if cfg0.calibration.threshold_strategy != "f1":
        raise SystemExit("calibration.threshold_strategy currently supports 'f1' only")

    method = cfg0.calibration.method
    y_val_true = y_val_true.astype(float).reshape(-1)
    y_val_score = y_val_score.astype(float).reshape(-1)

    if y_val_logits is not None:
        y_val_cal, params = calibrate_from_logits(y_val_true, y_val_logits, method=method)
    else:
        y_val_cal, params = calibrate_from_probs(y_val_true, y_val_score, method=method)

    thr_raw = select_threshold_f1(y_val_true, y_val_score)
    thr_cal = select_threshold_f1(y_val_true, y_val_cal)
    extra["val_best_threshold_raw_f1"] = float(thr_raw)
    extra["val_best_threshold_cal_f1"] = float(thr_cal)

    extra["val_logloss_raw"] = binary_log_loss(y_val_true, y_val_score)
    extra["val_brier_raw"] = binary_brier(y_val_true, y_val_score)
    extra["val_logloss_cal"] = binary_log_loss(y_val_true, y_val_cal)
    extra["val_brier_cal"] = binary_brier(y_val_true, y_val_cal)

    for k, v in params.items():
        extra[f"calibration_{k}"] = float(v)

    if not cfg0.calibration.use_calibrated:
        return y_test_score, extra

    if method == "temperature":
        t = float(params["temperature"])
        if y_test_logits is not None:
            y_test_cal = sigmoid(y_test_logits / t)
        else:
            eps = np.finfo(float).eps
            p = np.clip(y_test_score.astype(float), eps, 1.0 - eps)
            raw_logits = np.log(p / (1.0 - p))
            y_test_cal = sigmoid(raw_logits / t)
        return y_test_cal.astype(float), extra

    if method == "platt":
        a = float(params["a"])
        b = float(params["b"])
        if y_test_logits is not None:
            z = a * y_test_logits + b
        else:
            eps = np.finfo(float).eps
            p = np.clip(y_test_score.astype(float), eps, 1.0 - eps)
            raw_logits = np.log(p / (1.0 - p))
            z = a * raw_logits + b
        y_test_cal = sigmoid(z)
        return y_test_cal.astype(float), extra

    raise SystemExit(f"Unsupported calibration.method={method!r}")


def warn_unused_hpo_overrides(model_name: str, overrides: list[dict[str, object]]) -> None:
    valid_model_key = f"model.{model_name}."
    invalid_model_keys: set[str] = set()
    trainer_keys: set[str] = set()
    for override in overrides:
        for key in override.keys():
            if key.startswith("model.") and not key.startswith(valid_model_key):
                invalid_model_keys.add(key)
            if model_name == "xgboost" and key.startswith("trainer."):
                trainer_keys.add(key)
    if invalid_model_keys:
        print(
            "HPO overrides include keys not matching model "
            f"{model_name!r}: {sorted(invalid_model_keys)}",
            file=sys.stderr,
        )
    if trainer_keys:
        print(
            "HPO overrides include trainer.* keys, which are ignored for xgboost: "
            f"{sorted(trainer_keys)}",
            file=sys.stderr,
        )
