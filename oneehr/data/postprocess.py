from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FittedPostprocess:
    pipeline: list[dict[str, object]]


def _select_columns(X: pd.DataFrame, spec: str | list[str] | None) -> list[str]:
    if spec is None:
        return list(X.columns)
    if isinstance(spec, list):
        return [c for c in spec if c in X.columns]
    if not isinstance(spec, str):
        raise ValueError("step.cols must be a string, list[str], or null")

    if spec.endswith("*"):
        prefix = spec[:-1]
        return [c for c in X.columns if str(c).startswith(prefix)]
    if spec in {"num__*", "cat__*"}:
        prefix = spec[:-1]
        return [c for c in X.columns if str(c).startswith(prefix)]
    if spec in X.columns:
        return [spec]
    # Unknown single name => empty
    return []


def _ensure_numeric_frame(X: pd.DataFrame, cols: list[str]) -> None:
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"Columns must be numeric for this op: {non_numeric}")


def fit_postprocess_pipeline(X_train: pd.DataFrame, pipeline: list[dict[str, object]]) -> FittedPostprocess:
    """Fit post-merge preprocessing steps on training split only.

    The returned object can then be applied to train/val/test via `transform_postprocess_pipeline`.
    """

    X = X_train.copy()
    fitted_steps: list[dict[str, object]] = []

    for step in pipeline or []:
        if not isinstance(step, dict):
            raise ValueError("preprocess.pipeline steps must be TOML tables (dicts)")
        op = str(step.get("op", "")).lower().strip()
        if not op:
            raise ValueError("Each pipeline step must have non-empty `op`.")

        cols = _select_columns(X, step.get("cols"))
        if op == "standardize":
            _ensure_numeric_frame(X, cols)
            mean = X[cols].mean(axis=0, skipna=True)
            std = X[cols].std(axis=0, ddof=0, skipna=True).replace(0.0, 1.0)
            fitted_steps.append({"op": "standardize", "cols": cols, "mean": mean, "std": std})
            X.loc[:, cols] = (X[cols] - mean) / std
            continue

        if op == "impute":
            strategy = str(step.get("strategy", "mean")).lower()
            _ensure_numeric_frame(X, cols)

            if strategy == "mean":
                fill = X[cols].mean(axis=0, skipna=True)
                fill = fill.fillna(0.0)
            elif strategy == "median":
                fill = X[cols].median(axis=0, skipna=True)
                fill = fill.fillna(0.0)
            elif strategy == "constant":
                value = step.get("value", 0.0)
                fill = pd.Series({c: float(value) for c in cols})
            else:
                raise ValueError(f"Unsupported impute.strategy={strategy!r}")

            fitted_steps.append({"op": "impute", "cols": cols, "strategy": strategy, "fill": fill})
            X.loc[:, cols] = X[cols].fillna(fill)
            continue

        if op == "clip":
            _ensure_numeric_frame(X, cols)
            lower = step.get("lower")
            upper = step.get("upper")
            if lower is None and upper is None:
                raise ValueError("clip requires at least one of `lower` or `upper`.")
            fitted_steps.append({"op": "clip", "cols": cols, "lower": lower, "upper": upper})
            X.loc[:, cols] = X[cols].clip(lower=lower, upper=upper)
            continue

        if op == "winsorize":
            _ensure_numeric_frame(X, cols)
            lower_q = float(step.get("lower_q", 0.01))
            upper_q = float(step.get("upper_q", 0.99))
            if not (0.0 <= lower_q < upper_q <= 1.0):
                raise ValueError("winsorize requires 0 <= lower_q < upper_q <= 1")
            lo = X[cols].quantile(lower_q, interpolation="linear")
            hi = X[cols].quantile(upper_q, interpolation="linear")
            fitted_steps.append(
                {"op": "winsorize", "cols": cols, "lower_q": lower_q, "upper_q": upper_q, "lo": lo, "hi": hi}
            )
            X.loc[:, cols] = X[cols].clip(lower=lo, upper=hi, axis=1)
            continue

        raise ValueError(f"Unsupported preprocess.pipeline op={op!r}")

    return FittedPostprocess(pipeline=fitted_steps)


def transform_postprocess_pipeline(X: pd.DataFrame, fitted: FittedPostprocess) -> pd.DataFrame:
    X_out = X.copy()
    for step in fitted.pipeline:
        op = str(step["op"])
        cols = list(step.get("cols") or [])
        if not cols:
            continue
        if op == "standardize":
            mean = step["mean"]
            std = step["std"]
            X_out.loc[:, cols] = (X_out[cols] - mean) / std
        elif op == "impute":
            fill = step["fill"]
            X_out.loc[:, cols] = X_out[cols].fillna(fill)
        elif op == "clip":
            X_out.loc[:, cols] = X_out[cols].clip(lower=step.get("lower"), upper=step.get("upper"))
        elif op == "winsorize":
            X_out.loc[:, cols] = X_out[cols].clip(lower=step["lo"], upper=step["hi"], axis=1)
        else:
            raise ValueError(f"Unsupported fitted op={op!r}")
    return X_out


def maybe_fit_transform_postprocess(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame | None,
    X_test: pd.DataFrame | None,
    pipeline: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, FittedPostprocess | None]:
    if not pipeline:
        return X_train, X_val, X_test, None
    fitted = fit_postprocess_pipeline(X_train, pipeline)
    X_train_t = transform_postprocess_pipeline(X_train, fitted)
    X_val_t = None if X_val is None else transform_postprocess_pipeline(X_val, fitted)
    X_test_t = None if X_test is None else transform_postprocess_pipeline(X_test, fitted)
    return X_train_t, X_val_t, X_test_t, fitted

