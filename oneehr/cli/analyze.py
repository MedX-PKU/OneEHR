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
    """Feature importance for trained models (native for tree models, SHAP fallback)."""
    from oneehr.training.persistence import load_checkpoint

    train_dir = run_dir / "train"
    results = {}

    if not train_dir.exists():
        return {"module": "feature_importance", "models": {}}

    # Build tabular input from preprocessed data for SHAP background
    from oneehr.artifacts.manifest import read_manifest

    manifest = read_manifest(run_dir)
    feat_cols = manifest["feature_columns"]
    binned = pd.read_parquet(run_dir / "preprocess" / "binned.parquet")

    # Patient-level last-observation view
    if binned.empty:
        return {"module": "feature_importance", "models": {}}

    last = (
        binned.sort_values(["patient_id", "bin_time"], kind="stable")
        .groupby("patient_id", sort=False)[feat_cols]
        .last()
    )
    last.index = last.index.astype(str)

    # Join static features if present
    static_path = run_dir / "preprocess" / "static.parquet"
    if static_path.exists():
        static_df = pd.read_parquet(static_path)
        if "patient_id" in static_df.columns:
            static_df = static_df.set_index("patient_id")
        static_df.index = static_df.index.astype(str)
        overlap = [c for c in static_df.columns if c in last.columns]
        static_use = static_df.drop(columns=overlap, errors="ignore")
        last = last.join(static_use, how="left").fillna(0.0)

    for model_dir in sorted(train_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        ckpt_path = model_dir / "checkpoint.ckpt"
        if not ckpt_path.exists():
            continue
        model_name = model_dir.name

        try:
            model, meta = load_checkpoint(model_dir)
            stored_feat_cols = meta.get("feature_columns", feat_cols)
            X = last[stored_feat_cols]

            import torch
            if isinstance(model, torch.nn.Module):
                results[model_name] = {"skipped": "DL model — not supported yet"}
                continue

            # Try native importance first (XGBoost/CatBoost)
            from oneehr.analysis.feature_importance import xgboost_native_importance
            try:
                res = xgboost_native_importance(model, X, feature_names=stored_feat_cols)
                results[model_name] = {
                    "method": res.method,
                    "features": res.feature_names,
                    "importances": res.importances.tolist(),
                }
                continue
            except (TypeError, AttributeError):
                pass

            # Fallback: SHAP
            from oneehr.analysis.feature_importance import shap_importance
            res = shap_importance(
                model, X,
                task_kind=cfg.task.kind,
                feature_names=stored_feat_cols,
                nsamples=100,
            )
            results[model_name] = {
                "method": res.method,
                "features": res.feature_names,
                "importances": res.importances.tolist(),
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return {"module": "feature_importance", "models": results}
