"""oneehr analyze subcommand."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from oneehr.utils.io import ensure_dir, write_json


def run_analyze(
    cfg_path: str,
    run_dir: str | None,
    method: str | None,
) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.cli._common import resolve_run_root, require_manifest
    from oneehr.artifacts.run_io import RunIO
    from oneehr.analysis.feature_importance import (
        xgboost_native_importance,
        shap_importance,
        attention_importance,
    )

    cfg0 = load_experiment_config(cfg_path)
    run_root = resolve_run_root(cfg0, run_dir)
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}")

    manifest = require_manifest(run_root)

    feat_cols = manifest.dynamic_feature_columns()
    task_kind = str((manifest.data.get("task") or {}).get("kind", "binary"))

    from oneehr.models.constants import TABULAR_MODELS
    models_dir = run_root / "models"
    if not models_dir.exists():
        raise SystemExit(f"No models directory found at {models_dir}")

    run = RunIO(run_root=run_root)
    mode = str((manifest.data.get("task") or {}).get("prediction_mode", "patient"))
    if mode == "patient":
        X, y = run.load_patient_view(manifest)
    else:
        X, y, _ = run.load_time_view(manifest)

    analysis_dir = ensure_dir(run_root / "analysis")
    results_written = 0

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name

            if model_name in TABULAR_MODELS:
                from oneehr.models.tabular import load_tabular_model

                model_path = split_dir / "model.json"
                if not model_path.exists():
                    model_path = split_dir / "model.cbm"
                if not model_path.exists():
                    model_path = split_dir / "model.pkl"
                if not model_path.exists():
                    continue

                from oneehr.config.schema import TaskConfig

                task_cfg = TaskConfig(kind=task_kind, prediction_mode=mode)
                art = load_tabular_model(split_dir, task=task_cfg, kind=model_name)
                model_obj = art.model

                methods_to_run = []
                if method is not None:
                    methods_to_run = [method]
                else:
                    if model_name == "xgboost":
                        methods_to_run = ["xgboost", "shap"]
                    else:
                        methods_to_run = ["shap"]

                for m in methods_to_run:
                    if m == "xgboost" and model_name == "xgboost":
                        result = xgboost_native_importance(model_obj, X, feature_names=feat_cols)
                    elif m == "shap":
                        result = shap_importance(
                            model_obj, X,
                            task_kind=task_kind,
                            feature_names=feat_cols,
                            nsamples=min(500, len(X)),
                        )
                    else:
                        continue

                    out_path = analysis_dir / f"feature_importance_{model_name}_{split_name}_{m}.json"
                    write_json(out_path, {
                        "method": result.method,
                        "input_kind": result.input_kind,
                        "feature_names": result.feature_names,
                        "importances": result.importances.tolist(),
                    })
                    results_written += 1
                    print(f"Wrote {out_path.relative_to(run_root)}")

    if results_written == 0:
        print("No models found to analyze.", file=sys.stderr)
    else:
        print(f"Wrote {results_written} feature importance result(s) to {analysis_dir.relative_to(run_root)}/")
