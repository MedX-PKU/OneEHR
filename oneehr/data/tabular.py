"""Tabular views, postprocess pipeline, static features, and feature utilities.

Merged from: tabular.py, postprocess.py, static_postprocess.py, features.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─── Feature utilities ───────────────────────────────────────────────────────


def dynamic_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns whose names start with ``num__`` or ``cat__``."""
    return [c for c in df.columns if c.startswith("num__") or c.startswith("cat__")]


def has_static_branch(model: object) -> bool:
    """Check whether a model instance supports a dedicated static branch."""
    return hasattr(model, "static_dim") and int(getattr(model, "static_dim", 0)) > 0


# ─── Tabular views ───────────────────────────────────────────────────────────


def make_patient_tabular(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert binned long table into one-row-per-patient tabular features.

    Takes features from the last time bin (discretized) for each patient.
    """

    if "patient_id" not in binned.columns:
        raise ValueError("binned table must contain patient_id")
    if "label" not in binned.columns:
        feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
        if not feature_cols:
            raise ValueError("No feature columns found (expected num__/cat__ prefix)")
        binned = binned.sort_values(["patient_id", "bin_time"], kind="stable")
        X = binned.groupby("patient_id", sort=False)[feature_cols].last()
        y = pd.Series([], dtype=float, name="label")
        return X, y

    feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected num__/cat__ prefix)")

    binned = binned.sort_values(["patient_id", "bin_time"], kind="stable")
    X = binned.groupby("patient_id", sort=False)[feature_cols].last()
    y = binned.groupby("patient_id", sort=False)["label"].last()
    return X, y


def make_time_tabular(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return one-row-per-(patient_id, bin_time) tabular features."""

    required = {"patient_id", "bin_time"}
    missing = [c for c in required if c not in binned.columns]
    if missing:
        raise ValueError(f"binned missing columns: {missing}")

    feature_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    if not feature_cols:
        raise ValueError("No feature columns found (expected num__/cat__ prefix)")

    cols = ["patient_id", "bin_time", *feature_cols]
    if "label" in binned.columns:
        cols.insert(2, "label")
    df = binned[cols].copy().sort_values(["patient_id", "bin_time"], kind="stable")
    key = df[["patient_id", "bin_time"]].reset_index(drop=True)
    X = df[feature_cols].reset_index(drop=True)
    if "label" in df.columns:
        df = df.dropna(subset=["label"])
        key = df[["patient_id", "bin_time"]].reset_index(drop=True)
        X = df[feature_cols].reset_index(drop=True)
        y = df["label"].reset_index(drop=True)
    else:
        y = pd.Series([], dtype=float, name="label")
    return X, y, key


# ─── Postprocess pipeline ────────────────────────────────────────────────────


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
    return []


def _ensure_numeric_frame(X: pd.DataFrame, cols: list[str]) -> None:
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"Columns must be numeric for this op: {non_numeric}")


def _fit_fill_values(
    X: pd.DataFrame,
    cols: list[str],
    *,
    strategy: str,
    value: float | None = None,
) -> pd.Series:
    strategy = str(strategy).lower()
    if not cols:
        return pd.Series(dtype=float)

    _ensure_numeric_frame(X, cols)

    if strategy == "mean":
        fill = X[cols].mean(axis=0, skipna=True)
        return fill.fillna(0.0)
    if strategy == "median":
        fill = X[cols].median(axis=0, skipna=True)
        return fill.fillna(0.0)
    if strategy in {"mode", "most_frequent"}:
        out = {}
        for c in cols:
            s = X[c].dropna()
            if s.empty:
                out[c] = 0.0
                continue
            m = s.mode(dropna=True)
            out[c] = float(m.iloc[0]) if not m.empty else float(s.iloc[0])
        return pd.Series(out)
    if strategy == "constant":
        vv = 0.0 if value is None else float(value)
        return pd.Series({c: vv for c in cols})
    raise ValueError(f"Unsupported fill strategy={strategy!r}. Expected mean|median|mode|constant")


def fit_postprocess_pipeline(X_train: pd.DataFrame, pipeline: list[dict[str, object]]) -> FittedPostprocess:
    """Fit post-merge preprocessing steps on training split only."""

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
            if cols:
                X = X.copy()
                X[cols] = X[cols].astype(float)
            mean = X[cols].mean(axis=0, skipna=True)
            std = X[cols].std(axis=0, ddof=0, skipna=True).replace(0.0, 1.0)
            fitted_steps.append({"op": "standardize", "cols": cols, "mean": mean, "std": std})
            X.loc[:, cols] = (X[cols] - mean) / std
            continue

        if op == "impute":
            strategy = str(step.get("strategy", "mean")).lower()
            _ensure_numeric_frame(X, cols)
            fill = _fit_fill_values(X, cols, strategy=strategy, value=step.get("value"))
            fitted_steps.append({"op": "impute", "cols": cols, "strategy": strategy, "fill": fill})
            X.loc[:, cols] = X[cols].fillna(fill)
            continue

        if op == "forward_fill":
            group_key = str(step.get("group_key", "patient_id"))
            order_key = str(step.get("order_key", "bin_time"))
            fallback = step.get("fallback") or {}
            if not isinstance(fallback, dict):
                raise ValueError("forward_fill.fallback must be a table (dict)")

            fallback_strategy = str(fallback.get("strategy", "mean")).lower()
            fallback_value = fallback.get("value")

            if group_key not in X.columns or order_key not in X.columns:
                raise ValueError(
                    f"forward_fill requires X to include columns {group_key!r} and {order_key!r}. "
                    "Use forward_fill only for time-level features with explicit time ordering."
                )

            ff_cols = [c for c in cols if c not in {group_key, order_key}]
            _ensure_numeric_frame(X, ff_cols)
            fill = _fit_fill_values(X, ff_cols, strategy=fallback_strategy, value=fallback_value)
            fitted_steps.append(
                {
                    "op": "forward_fill",
                    "cols": ff_cols,
                    "group_key": group_key,
                    "order_key": order_key,
                    "fallback": {"strategy": fallback_strategy, "value": fallback_value},
                    "fill": fill,
                }
            )
            X = X.sort_values([group_key, order_key], kind="stable")
            X.loc[:, ff_cols] = X.groupby(group_key, sort=False)[ff_cols].ffill()
            X.loc[:, ff_cols] = X[ff_cols].fillna(fill)
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
            X_out = X_out.copy()
            X_out[cols] = X_out[cols].apply(pd.to_numeric, errors="coerce").astype(float)
            mean_s = pd.Series(mean, dtype=float) if isinstance(mean, dict) else pd.Series(mean).astype(float)
            std_s = (
                pd.Series(std, dtype=float).replace(0.0, 1.0)
                if isinstance(std, dict)
                else pd.Series(std).astype(float).replace(0.0, 1.0)
            )
            mean_s = mean_s.reindex(cols)
            std_s = std_s.reindex(cols)
            X_out[cols] = ((X_out[cols] - mean_s) / std_s).astype(float)
        elif op == "impute":
            fill = step["fill"]
            X_out.loc[:, cols] = X_out[cols].fillna(fill)
        elif op == "forward_fill":
            group_key = str(step["group_key"])
            order_key = str(step["order_key"])
            fill = step["fill"]
            if group_key not in X_out.columns or order_key not in X_out.columns:
                raise ValueError(
                    f"forward_fill requires columns {group_key!r} and {order_key!r} in X."
                )
            X_out = X_out.sort_values([group_key, order_key], kind="stable")
            X_out.loc[:, cols] = X_out.groupby(group_key, sort=False)[cols].ffill()
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


# ─── Static feature processing ───────────────────────────────────────────────


@dataclass(frozen=True)
class StaticArtifacts:
    """Artifacts needed to reproduce static feature processing."""

    schema_version: int
    raw_cols: list[str]
    feature_columns: list[str]
    feature_columns_sha256: str
    fitted_postprocess: FittedPostprocess | None


def _encode_static_categoricals(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw static columns into numeric columns compatible with postprocess."""

    if raw.empty:
        return raw.copy()

    id_like_cols = [c for c in raw.columns if str(c).lower() in {"patient_id", "patientid"}]
    if id_like_cols:
        raw = raw.drop(columns=id_like_cols, errors="ignore")

    X_parts: list[pd.DataFrame] = []

    for col in raw.columns:
        s = raw[col]
        if pd.api.types.is_numeric_dtype(s):
            X_parts.append(pd.DataFrame({f"num__{col}": pd.to_numeric(s, errors="coerce")}))
            continue

        s2 = s.astype("string")
        s2 = s2.fillna("__nan__")
        d = pd.get_dummies(s2, prefix=f"cat__{col}", prefix_sep="__", dtype=float)
        X_parts.append(d)

    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=raw.index)
    X.index = raw.index
    return X


def fit_transform_static_features(
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame | None,
    raw_test: pd.DataFrame | None,
    *,
    pipeline: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, StaticArtifacts]:
    """Fit static feature encoder + postprocess on train only, then transform all splits."""
    from oneehr.utils import sha256_lines

    X_train0 = _encode_static_categoricals(raw_train)
    X_val0 = None if raw_val is None else _encode_static_categoricals(raw_val)
    X_test0 = None if raw_test is None else _encode_static_categoricals(raw_test)

    cols = list(X_train0.columns)
    if X_val0 is not None:
        cols = sorted(set(cols).union(X_val0.columns))
    if X_test0 is not None:
        cols = sorted(set(cols).union(X_test0.columns))

    X_train0 = X_train0.reindex(columns=cols).fillna(0.0)
    X_val0 = None if X_val0 is None else X_val0.reindex(columns=cols).fillna(0.0)
    X_test0 = None if X_test0 is None else X_test0.reindex(columns=cols).fillna(0.0)

    X_train, X_val, X_test, fitted_post = maybe_fit_transform_postprocess(
        X_train=X_train0,
        X_val=X_val0,
        X_test=X_test0,
        pipeline=pipeline,
    )

    feat_cols = list(X_train.columns)
    artifacts = StaticArtifacts(
        schema_version=1,
        raw_cols=list(raw_train.columns),
        feature_columns=feat_cols,
        feature_columns_sha256=sha256_lines(feat_cols),
        fitted_postprocess=fitted_post,
    )
    return X_train, X_val, X_test, artifacts
