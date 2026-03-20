"""oneehr analyze subcommand.

Reads test/predictions.parquet and produces analyze/{module}.json files.
"""
from __future__ import annotations

import json
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
        "fairness": _run_fairness,
        "calibration": _run_calibration,
        "statistical_tests": _run_statistical_tests,
        "missing_data": _run_missing_data,
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
        if isinstance(result, tuple):
            # Modules that produce extra artifacts (e.g. calibration)
            result_dict, extra_df = result
            write_json(analyze_dir / f"{name}.json", result_dict)
            if extra_df is not None and not extra_df.empty:
                extra_df.to_parquet(analyze_dir / f"calibrated_predictions.parquet", index=False)
                print(f"  Wrote {analyze_dir / 'calibrated_predictions.parquet'}")
        else:
            write_json(analyze_dir / f"{name}.json", result)
        print(f"  Wrote {analyze_dir / name}.json")


def _run_comparison(*, preds: pd.DataFrame, cfg, run_dir: Path) -> dict:
    """Cross-system comparison metrics with bootstrap confidence intervals."""
    from oneehr.eval.bootstrap import bootstrap_metric
    from oneehr.eval.metrics import binary_metrics, regression_metrics

    # Key metrics to bootstrap per task kind
    ci_metrics = {
        "binary": ["auroc", "auprc", "f1"],
        "regression": ["mae", "rmse", "r2"],
    }

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

        # Bootstrap CIs for key metrics
        for m in ci_metrics.get(cfg.task.kind, []):
            if m not in metrics:
                continue
            try:
                br = bootstrap_metric(
                    y_true=y_true, y_pred=y_pred,
                    task=cfg.task, metric=m,
                    n=1000, seed=42, ci=0.95,
                )
                metrics[f"{m}_ci_low"] = br.ci_low
                metrics[f"{m}_ci_high"] = br.ci_high
            except Exception:
                pass

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
    """Feature importance for trained models (native for tree, SHAP fallback, IG for DL)."""
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
                # Use Integrated Gradients for DL models
                try:
                    from oneehr.analysis.feature_importance import integrated_gradients_importance
                    from oneehr.data.sequence import build_patient_sequences, pad_sequences

                    patient_ids, seqs, lengths = build_patient_sequences(binned, stored_feat_cols)
                    X_padded = pad_sequences(seqs, lengths)
                    res = integrated_gradients_importance(
                        model, X_padded, lengths,
                        feature_names=stored_feat_cols,
                        n_steps=50,
                        max_patients=200,
                    )
                    results[model_name] = {
                        "method": res.method,
                        "features": res.feature_names,
                        "importances": res.importances.tolist(),
                    }
                except Exception as e:
                    results[model_name] = {"error": f"IG failed: {e}"}
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


def _run_fairness(*, preds: pd.DataFrame, cfg, run_dir: Path) -> dict:
    """Fairness/bias analysis across sensitive attributes."""
    if cfg.task.kind != "binary":
        return {"module": "fairness", "note": "only supported for binary tasks"}

    static_path = run_dir / "preprocess" / "static.parquet"
    if not static_path.exists():
        return {"module": "fairness", "note": "no static.parquet found"}

    static = pd.read_parquet(static_path)
    from oneehr.analysis.fairness import compute_fairness

    result = compute_fairness(preds=preds, static=static)
    return {"module": "fairness", **result}


def _run_calibration(*, preds: pd.DataFrame, cfg, run_dir: Path) -> tuple[dict, pd.DataFrame | None]:
    """Post-hoc calibration (temperature, Platt, isotonic)."""
    if cfg.task.kind != "binary":
        return {"module": "calibration", "note": "only supported for binary tasks"}, None

    split_path = run_dir / "preprocess" / "split.json"
    if not split_path.exists():
        return {"module": "calibration", "note": "no split.json found"}, None

    split_info = json.loads(split_path.read_text(encoding="utf-8"))
    from oneehr.analysis.calibration import compute_calibration

    return compute_calibration(preds=preds, split_info=split_info)


def _run_statistical_tests(*, preds: pd.DataFrame, cfg, run_dir: Path) -> dict:
    """Pairwise statistical tests (DeLong, McNemar) between systems."""
    if cfg.task.kind != "binary":
        return {"module": "statistical_tests", "note": "only supported for binary tasks"}

    from oneehr.analysis.statistical_tests import compute_statistical_tests

    return compute_statistical_tests(preds=preds)


def _run_missing_data(*, preds: pd.DataFrame, cfg, run_dir: Path) -> dict:
    """Missing data quality report from preprocessed data."""
    binned_path = run_dir / "preprocess" / "binned.parquet"
    if not binned_path.exists():
        return {"module": "missing_data", "note": "no binned.parquet found"}

    binned = pd.read_parquet(binned_path)
    from oneehr.analysis.missing_data import compute_missing_data

    return compute_missing_data(binned=binned)
