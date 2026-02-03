from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


InputKind = Literal["tabular", "sequence"]


@dataclass(frozen=True)
class FeatureImportanceResult:
    """Feature importance scores.

    `importances` is a 1D array aligned with `feature_names`.
    `details` is method-specific extra information (e.g. SHAP raw arrays).
    """

    method: str
    input_kind: InputKind
    feature_names: list[str]
    importances: np.ndarray
    details: dict[str, Any] | None = None


def _as_2d_tabular_input(
    X: pd.DataFrame | np.ndarray | "torch.Tensor",
    *,
    feature_names: list[str] | None = None,
    time_axis: int = 1,
) -> tuple[np.ndarray, list[str], InputKind]:
    """Normalize input to a 2D numpy array (N, D).

    Supports:
    - Tabular: DataFrame (N,D) / ndarray (N,D)
    - Sequence: ndarray (N,T,D) -> take last time step
    - Sequence: torch.Tensor (N,T,D) -> take last time step

    For sequence inputs used with tabular models (e.g. XGBoost),
    this function applies the repo convention: use the last time step.
    """

    # Avoid importing torch unless needed.
    torch = None
    if "torch" in str(type(X)):
        try:
            import torch as _torch  # type: ignore

            torch = _torch
        except ModuleNotFoundError:
            torch = None

    input_kind: InputKind = "tabular"
    if isinstance(X, pd.DataFrame):
        feats = list(X.columns) if feature_names is None else feature_names
        return X.to_numpy(), feats, input_kind

    if torch is not None and isinstance(X, torch.Tensor):
        x_np = X.detach().cpu().numpy()
        if x_np.ndim == 3:
            input_kind = "sequence"
            x_np = np.take(x_np, indices=[-1], axis=time_axis).squeeze(axis=time_axis)
        elif x_np.ndim != 2:
            raise ValueError(f"Unsupported torch tensor shape: {tuple(x_np.shape)}")
        feats = feature_names if feature_names is not None else [f"f{i}" for i in range(x_np.shape[1])]
        return x_np, feats, input_kind

    x_np = np.asarray(X)
    if x_np.ndim == 3:
        input_kind = "sequence"
        x_np = np.take(x_np, indices=[-1], axis=time_axis).squeeze(axis=time_axis)
    elif x_np.ndim != 2:
        raise ValueError(f"Unsupported input shape: {tuple(x_np.shape)}")

    feats = feature_names if feature_names is not None else [f"f{i}" for i in range(x_np.shape[1])]
    return x_np, feats, input_kind


def xgboost_native_importance(
    model: Any,
    X: pd.DataFrame | np.ndarray | "torch.Tensor",
    *,
    importance_type: str = "gain",
    feature_names: list[str] | None = None,
) -> FeatureImportanceResult:
    """Extract feature importance from an XGBoost sklearn wrapper model.

    If X is sequence-shaped (N,T,D), uses the last time step (T-1).
    """

    x_np, feats, input_kind = _as_2d_tabular_input(X, feature_names=feature_names)
    _ = x_np  # only used for shape normalization / feature names

    # XGBClassifier/XGBRegressor expose `feature_importances_`.
    # Booster-based access gives you choice of importance_type.
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        b = model.get_booster()
        score = b.get_score(importance_type=importance_type)
        # Keys are like "f0", "f1" unless feature_names were provided at fit time.
        values = np.zeros(len(feats), dtype=float)
        for i, name in enumerate(feats):
            values[i] = float(score.get(name, score.get(f"f{i}", 0.0)))
        return FeatureImportanceResult(
            method=f"xgboost_native:{importance_type}",
            input_kind=input_kind,
            feature_names=feats,
            importances=values,
            details={"raw_score": score},
        )

    fi = getattr(model, "feature_importances_", None)
    if fi is None:
        raise TypeError("Model does not expose get_booster() or feature_importances_.")
    values = np.asarray(fi, dtype=float)
    if values.shape != (len(feats),):
        values = values.reshape(-1)
    return FeatureImportanceResult(
        method="xgboost_native:feature_importances_",
        input_kind=input_kind,
        feature_names=feats,
        importances=values,
        details=None,
    )


def shap_importance(
    model: Any,
    X: pd.DataFrame | np.ndarray | "torch.Tensor",
    *,
    task_kind: Literal["binary", "regression"] | None = None,
    feature_names: list[str] | None = None,
    nsamples: int | None = None,
    seed: int = 0,
) -> FeatureImportanceResult:
    """Compute global feature importance via mean(|SHAP value|).

    Notes:
    - For sequence inputs used with tabular models, uses last time step.
    - For binary models that output probabilities, prefer passing a function that
      returns raw scores/logits if you want additive explanations; otherwise SHAP
      will explain the probability output.
    """

    import shap  # hard dependency (per project decision)

    x_np, feats, input_kind = _as_2d_tabular_input(X, feature_names=feature_names)

    if nsamples is not None and x_np.shape[0] > nsamples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x_np.shape[0], size=nsamples, replace=False)
        x_used = x_np[idx]
    else:
        x_used = x_np

    # Use the most general-purpose interface: shap.Explainer.
    # It will choose TreeExplainer/LinearExplainer/etc when possible.
    # Prefer disabling the numba acceleration path because this repo targets
    # Python 3.12, and SHAP's numba stack is often a version constraint hotspot.
    # This keeps `shap` usable even when `numba` isn't available.
    explainer = shap.Explainer(model, x_used, use_numba=False)
    exp = explainer(x_used)

    values = exp.values
    # values can be:
    # - (N, D) regression or single-output
    # - (N, D, C) classification (depending on model/explainer)
    values_np = np.asarray(values)
    if values_np.ndim == 3:
        # For binary classification, SHAP may return 2 classes. If task_kind is
        # provided and binary, focus on the positive class.
        if values_np.shape[2] == 2 and task_kind == "binary":
            values_np = values_np[:, :, 1]
        else:
            # Otherwise aggregate across outputs/classes.
            values_np = np.abs(values_np).mean(axis=2)

    global_imp = np.abs(values_np).mean(axis=0)
    return FeatureImportanceResult(
        method="shap:mean_abs",
        input_kind=input_kind,
        feature_names=feats,
        importances=np.asarray(global_imp, dtype=float),
        details={"shap_values": values_np},
    )


def attention_importance(
    attn_weights: np.ndarray,
    X: np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> FeatureImportanceResult:
    """Compute feature importance using attention over time.

    This function assumes:
    - `attn_weights`: (B, T) attention distribution over time steps
    - `X`: (B, T, D) input features

    Returns importance per feature by weighting |X| with attention and averaging
    over batch and time.
    """

    if attn_weights.ndim != 2:
        raise ValueError(f"attn_weights must be (B,T), got {tuple(attn_weights.shape)}")
    if X.ndim != 3:
        raise ValueError(f"X must be (B,T,D), got {tuple(X.shape)}")
    if attn_weights.shape[0] != X.shape[0] or attn_weights.shape[1] != X.shape[1]:
        raise ValueError(
            "attn_weights shape must match X batch/time dims: "
            f"attn={tuple(attn_weights.shape)} X={tuple(X.shape)}"
        )

    B, T, D = X.shape
    feats = feature_names if feature_names is not None else [f"f{i}" for i in range(D)]
    if len(feats) != D:
        raise ValueError("feature_names length must match X.shape[2]")

    w = attn_weights / (attn_weights.sum(axis=1, keepdims=True) + 1e-12)
    w = w[:, :, None]  # (B,T,1)
    imp = (np.abs(X) * w).sum(axis=1).mean(axis=0)  # (D,)
    return FeatureImportanceResult(
        method="attention:abs_x_weighted",
        input_kind="sequence",
        feature_names=feats,
        importances=np.asarray(imp, dtype=float),
        details=None,
    )
