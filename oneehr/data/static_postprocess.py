from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneehr.data.postprocess import FittedPostprocess, maybe_fit_transform_postprocess


@dataclass(frozen=True)
class StaticArtifacts:
    """Artifacts needed to reproduce static feature processing."""

    schema_version: int
    raw_cols: list[str]
    feature_columns: list[str]
    feature_columns_sha256: str
    fitted_postprocess: FittedPostprocess | None


def _sha256_lines(lines: list[str]) -> str:
    import hashlib

    norm = "\n".join([ln.strip() for ln in lines]) + "\n"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _encode_static_categoricals(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw static columns into numeric columns compatible with postprocess.

    Strategy (MVP):
    - Numeric columns => `num__{col}` float
    - Non-numeric columns => one-hot via `pd.get_dummies` with `cat__{col}__{level}`
    - ID columns like `patient_id` are not features and must be removed upstream
    - Missing numeric stays NaN (handled by postprocess.impute if configured)
    - Missing categorical => treated as its own level via `__nan__`
    """

    if raw.empty:
        return raw.copy()

    # Defensive: `patient_id` must never become a feature column (leakage + huge cardinality).
    # Drop common ID-like variants rather than erroring.
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

    X_train0 = _encode_static_categoricals(raw_train)
    X_val0 = None if raw_val is None else _encode_static_categoricals(raw_val)
    X_test0 = None if raw_test is None else _encode_static_categoricals(raw_test)

    # Align columns across splits pre-postprocess.
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
        feature_columns_sha256=_sha256_lines(feat_cols),
        fitted_postprocess=fitted_post,
    )
    return X_train, X_val, X_test, artifacts
