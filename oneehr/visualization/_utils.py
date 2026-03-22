"""Shared utility functions for OneEHR visualization modules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load test/predictions.parquet from a run directory."""
    path = run_dir / "test" / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No predictions found at {path}")
    return pd.read_parquet(path)


def load_analysis_json(run_dir: Path, module: str) -> dict:
    """Load analyze/{module}.json from a run directory."""
    path = run_dir / "analyze" / f"{module}.json"
    if not path.exists():
        raise FileNotFoundError(f"No analysis output at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_meta(run_dir: Path, model_name: str) -> dict:
    """Load train/{model_name}/meta.json."""
    path = run_dir / "train" / model_name / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No training metadata at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(run_dir: Path) -> dict:
    """Load preprocess/split.json."""
    path = run_dir / "preprocess" / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"No split at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def system_predictions(
    preds: pd.DataFrame,
    system: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (y_true, y_pred) arrays for a single system, dropping NaNs."""
    sdf = preds[preds["system"] == system]
    y_true = sdf["y_true"].to_numpy(dtype=float)
    y_pred = sdf["y_pred"].to_numpy(dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[finite], y_pred[finite]


def bootstrap_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    curve_fn,
    *,
    n_boot: int = 200,
    seed: int = 42,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence bands for a curve (ROC or PR).

    Parameters
    ----------
    curve_fn : callable
        Function(y_true, y_pred) -> (x_vals, y_vals).  For ROC this is
        (fpr, tpr); for PR this is (recall, precision).
    n_points : int
        Number of interpolation points for the common x-axis.

    Returns
    -------
    x_common : (n_points,)
    y_low : (n_points,) lower 2.5% band
    y_high : (n_points,) upper 97.5% band
    """
    rng = np.random.RandomState(seed)
    x_common = np.linspace(0.0, 1.0, n_points)
    y_boots = np.empty((n_boot, n_points))

    n = len(y_true)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            y_boots[i] = np.nan
            continue
        xv, yv = curve_fn(yt, yp)
        y_boots[i] = np.interp(x_common, xv, yv)

    y_low = np.nanpercentile(y_boots, 2.5, axis=0)
    y_high = np.nanpercentile(y_boots, 97.5, axis=0)
    return x_common, y_low, y_high
