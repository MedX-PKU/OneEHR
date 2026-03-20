"""oneehr analyze subcommand.

Reads test/predictions.parquet and produces analyze/{module}.json files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.utils import ensure_dir, write_json


def run_analyze(cfg_path: str, *, module: str | None = None) -> None:
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(cfg_path)
    run_dir = cfg.run_dir()

    preds_path = run_dir / "test" / "predictions.parquet"
    if not preds_path.exists():
        raise SystemExit(
            f"No predictions found at {preds_path}. Run `oneehr test` first."
        )

    preds = pd.read_parquet(preds_path)
    analyze_dir = ensure_dir(run_dir / "analyze")

    # Available modules
    available = {
        "comparison": _run_comparison,
        "feature_importance": _run_feature_importance,
    }

    if module is not None:
        if module not in available:
            raise SystemExit(
                f"Unknown analysis module: {module!r}. "
                f"Available: {sorted(available.keys())}"
            )
        modules_to_run = {module: available[module]}
    else:
        modules_to_run = available

    for name, fn in modules_to_run.items():
        print(f"Running analysis module: {name}")
        result = fn(preds=preds, cfg=cfg, run_dir=run_dir)
        write_json(analyze_dir / f"{name}.json", result)
        print(f"  Wrote {analyze_dir / name}.json")


def _run_comparison(*, preds: pd.DataFrame, cfg, run_dir: Path) -> dict:
    """Cross-system comparison metrics."""
    from oneehr.eval.metrics import binary_metrics, regression_metrics

    systems = []
    for system_name in preds["system"].unique():
        sdf = preds[preds["system"] == system_name]
        y_true = sdf["y_true"].to_numpy(dtype=float)
        y_pred = sdf["y_pred"].to_numpy(dtype=float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[finite], y_pred[finite]

        if y_true.size == 0:
            systems.append({"name": system_name, "n": 0, "metrics": {}})
            continue

        if cfg.task.kind == "binary":
            metrics = binary_metrics(y_true, y_pred).metrics
        else:
            metrics = regression_metrics(y_true, y_pred).metrics

        systems.append({
            "name": system_name,
            "n": int(y_true.size),
            "metrics": metrics,
        })

    return {
        "module": "comparison",
        "task": {"kind": cfg.task.kind, "prediction_mode": cfg.task.prediction_mode},
        "systems": systems,
    }


def _run_feature_importance(*, preds: pd.DataFrame, cfg, run_dir: Path) -> dict:
    """SHAP-based feature importance for trained models."""
    train_dir = run_dir / "train"
    results = {}

    if not train_dir.exists():
        return {"module": "feature_importance", "models": {}}

    for model_dir in sorted(train_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        try:
            from oneehr.analysis.feature_importance import compute_shap_importance
            importance = compute_shap_importance(
                model_dir=model_dir,
                run_dir=run_dir,
                feat_cols=None,  # will read from meta.json
            )
            results[model_name] = importance
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return {"module": "feature_importance", "models": results}
