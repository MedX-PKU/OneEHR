"""oneehr train subcommand."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from oneehr.eval.calibration import sigmoid
from oneehr.models import TABULAR_MODELS, DL_MODELS
from oneehr.data.tabular import has_static_branch
from oneehr.cli._train_eval import maybe_calibrate_and_threshold, warn_unused_hpo_overrides
from oneehr.cli._train_dl import train_dl_patient_level, train_dl_time_level
from oneehr.utils import ensure_dir, write_json


def run_train(cfg_path: str, force: bool) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.artifacts.store import RunIO

    cfg0 = load_experiment_config(cfg_path)
    _require_training_model(cfg0)
    out_root = cfg0.output.root / cfg0.output.run_name

    existing_training_artifacts = [
        "models",
        "preds",
        "hpo",
        "summary.json",
        "hpo_best.csv",
        "preprocess",
    ]
    has_existing_training_outputs = any((out_root / rel).exists() for rel in existing_training_artifacts)
    if has_existing_training_outputs and not force:
        raise SystemExit(
            f"Training artifacts already exist under {out_root}. "
            "Pass --force to overwrite them."
        )
    ensure_dir(out_root)

    run = RunIO(run_root=out_root)
    _ = run.require_manifest()

    if force:
        import shutil

        for rel in existing_training_artifacts:
            p = out_root / rel
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()

    _run_benchmark(cfg_path, force=force)


def _run_benchmark(cfg_path: str, *, force: bool = False) -> None:
    import pandas as pd

    from oneehr.config.load import load_experiment_config
    from oneehr.data.io import load_dynamic_table_optional, load_static_table
    from oneehr.data.splits import require_saved_splits, _parse_repeat_index
    from oneehr.data.tabular import maybe_fit_transform_postprocess
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.hpo.grid import apply_overrides, iter_grid
    from oneehr.models import build_model
    from oneehr.models.tree import (
        predict_tabular,
        predict_tabular_logits,
        save_tabular_model,
        train_tabular_model,
    )
    from oneehr.modeling.persistence import write_dl_artifacts
    from oneehr.modeling.trainer import fit_model
    from oneehr.artifacts.store import RunIO
    from oneehr.cli._train_hpo import run_single_hpo, run_cv_mean_hpo, run_per_split_hpo

    cfg0 = load_experiment_config(cfg_path)
    primary_model = _require_training_model(cfg0)
    train_dataset = cfg0.datasets.train if cfg0.datasets is not None else cfg0.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static_raw = load_static_table(train_dataset.static)
    out_root = cfg0.output.root / cfg0.output.run_name
    splits = require_saved_splits(out_root / "splits", context="running `oneehr train`")
    run = RunIO(run_root=out_root)
    manifest = run.require_manifest()
    labels_df = run.load_labels(manifest)
    binned = run.load_binned(manifest)

    if cfg0.task.prediction_mode == "patient":
        X, y = run.load_patient_view(manifest)
        key = None
        global_mask = None
    elif cfg0.task.prediction_mode == "time":
        X, y, key = run.load_time_view(manifest)
        global_mask = None
    else:
        raise SystemExit(f"Unsupported task.prediction_mode={cfg0.task.prediction_mode!r}")
    dynamic_feature_columns = manifest.dynamic_feature_columns()

    static_all = None
    static_feature_columns = None
    st_path = manifest.static_matrix_path()
    if st_path is not None:
        static_all = pd.read_parquet(out_root / st_path)
        static_feature_columns = manifest.static_feature_columns()
        if list(static_all.columns) != list(static_feature_columns):
            raise SystemExit(
                "Static feature_columns mismatch with run_manifest.json. "
                "Re-run `oneehr preprocess`."
            )

    models = cfg0.models or [primary_model]

    rows = []
    run_records: list[dict[str, object]] = []

    for model_cfg in models:
        cfg_model = replace(cfg0, model=model_cfg, models=[model_cfg])
        model_name = cfg_model.model.name
        if model_name in cfg0.hpo_by_model:
            cfg_model = replace(cfg_model, hpo=cfg0.hpo_by_model[model_name])

        warn_unused_hpo_overrides(model_name, list(iter_grid(cfg_model.hpo)))

        # --- HPO (single / cv_mean scope) ---
        reused_overrides: dict[str, object] | None = None
        if cfg_model.hpo.enabled and cfg_model.hpo.scope == "single":
            reused_overrides = run_single_hpo(
                cfg_model, model_name, X, y, key, global_mask, splits, out_root,
                binned=binned, labels_df=labels_df,
                dynamic_feature_columns=dynamic_feature_columns,
            )

        if cfg_model.hpo.enabled and cfg_model.hpo.scope == "cv_mean":
            reused_overrides = run_cv_mean_hpo(
                cfg_model, model_name, X, y, key, global_mask, splits, out_root,
            )

        # --- Per-split training ---
        for sp in splits:
            test_key = None
            if cfg_model.task.prediction_mode == "patient":
                train_mask = X.index.astype(str).isin(sp.train_patients)
                val_mask = X.index.astype(str).isin(sp.val_patients)
                test_mask = X.index.astype(str).isin(sp.test_patients)
            else:
                assert key is not None
                train_mask = key["patient_id"].astype(str).isin(sp.train_patients)
                val_mask = key["patient_id"].astype(str).isin(sp.val_patients)
                test_mask = key["patient_id"].astype(str).isin(sp.test_patients)

            if global_mask is not None:
                train_mask2 = train_mask & global_mask
                val_mask2 = val_mask & global_mask
                test_mask2 = test_mask & global_mask
            else:
                train_mask2 = train_mask
                val_mask2 = val_mask
                test_mask2 = test_mask

            X_train, y_train = X.loc[train_mask2], y.loc[train_mask2].to_numpy()
            X_val, y_val = X.loc[val_mask2], y.loc[val_mask2].to_numpy()
            X_test, y_test = X.loc[test_mask2], y.loc[test_mask2].to_numpy()
            if cfg_model.task.prediction_mode == "patient":
                keep_tr = ~np.isnan(y_train.astype(float))
                keep_va = ~np.isnan(y_val.astype(float))
                keep_te = ~np.isnan(y_test.astype(float))
                X_train, y_train = X_train.iloc[keep_tr], y_train[keep_tr]
                X_val, y_val = X_val.iloc[keep_va], y_val[keep_va]
                X_test, y_test = X_test.iloc[keep_te], y_test[keep_te]
            if cfg_model.task.prediction_mode == "time":
                assert key is not None
                test_key = key.loc[test_mask2].reset_index(drop=True)
                test_patient_ids = test_key["patient_id"].astype(str)
            else:
                test_patient_ids = X_test.index.astype(str)

            static_only = len(dynamic_feature_columns) == 0
            if static_only:
                raise SystemExit(
                    "Static-only datasets are not supported in this version. "
                    "Provide dynamic features."
                )

            X_train, X_val, X_test, fitted_post = maybe_fit_transform_postprocess(
                X_train=X_train, X_val=X_val, X_test=X_test,
                pipeline=cfg0.preprocess.pipeline,
            )

            static_train = None
            if static_all is not None and static_feature_columns is not None:
                static_train = static_all
            if static_all is not None and static_feature_columns is not None and model_name in TABULAR_MODELS:
                overlap = [c for c in static_feature_columns if c in X_train.columns]
                static_use = static_all.drop(columns=overlap, errors="ignore")
                X_train = X_train.join(static_use, how="left").fillna(0.0)
                X_val = X_val.join(static_use, how="left").fillna(0.0)
                X_test = X_test.join(static_use, how="left").fillna(0.0)

            # --- Per-split HPO ---
            if cfg_model.hpo.enabled and cfg_model.hpo.scope == "per_split":
                best_overrides, hpo_best_score = run_per_split_hpo(
                    cfg_model, model_name, sp,
                    X, y, key, global_mask, out_root,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    binned=binned, labels_df=labels_df,
                    dynamic_feature_columns=dynamic_feature_columns,
                )
                cfg = apply_overrides(cfg_model, best_overrides)
                hpo_best_str = str(best_overrides)
            else:
                if reused_overrides is not None:
                    cfg = apply_overrides(cfg_model, reused_overrides)
                    hpo_best_str = str(reused_overrides)
                else:
                    cfg = cfg_model
                    hpo_best_str = ""
                hpo_best_score = None

            # ===== Train on this split =====
            # Determine effective seed for multi-seed training.
            repeat_idx = _parse_repeat_index(sp.name)
            effective_seed = cfg.trainer.seed + repeat_idx

            if cfg.model.name in TABULAR_MODELS:
                if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                    rows.append({"model": model_name, "split": sp.name, "skipped": 1, "reason": "single_class_train"})
                    continue

                tab_model_cfg = getattr(cfg.model, cfg.model.name)
                art = train_tabular_model(
                    model_name=cfg.model.name,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    task=cfg.task, model_cfg=tab_model_cfg,
                    seed=effective_seed,
                )
                model_out = ensure_dir(out_root / "models" / model_name / sp.name)
                save_tabular_model(art, model_out)
                y_score = predict_tabular(art, X_test, cfg.task)
                if fitted_post is not None:
                    pp_dir = ensure_dir(out_root / "preprocess" / sp.name)
                    write_json(pp_dir / "pipeline.json", {"pipeline": fitted_post.pipeline})
            else:
                feat_cols = list(dynamic_feature_columns)
                input_dim = len(feat_cols)
                if cfg.task.kind == "binary" and len(np.unique(y_train.astype(float))) < 2:
                    rows.append({"model": model_name, "split": sp.name, "skipped": 1, "reason": "single_class_train"})
                    continue
                if cfg.task.kind == "binary" and len(np.unique(y_val.astype(float))) < 2:
                    rows.append({"model": model_name, "split": sp.name, "skipped": 1, "reason": "single_class_val"})
                    continue

                cfg_use = replace(cfg, _dynamic_dim=input_dim)
                built = build_model(cfg_use)
                model = built.model

                model_supports_static_branch = has_static_branch(model)

                trainer_cfg_eff = replace(cfg.trainer, seed=effective_seed)
                if cfg.task.prediction_mode == "patient":
                    (
                        y_score, y_test, test_patient_ids, test_logits,
                        val_score, val_logits, y_val_true,
                    ) = train_dl_patient_level(
                        model=model, binned=binned, y=y,
                        static=static_train, split=sp,
                        cfg=trainer_cfg_eff, task=cfg.task,
                        model_supports_static_branch=model_supports_static_branch,
                        y_map=dict(zip(X.index.astype(str).tolist(), y.to_numpy().tolist())),
                    )
                else:
                    (
                        y_score, y_test, test_key_rows, test_logits,
                        val_score, val_logits, y_val_true,
                    ) = train_dl_time_level(
                        model=model, binned=binned, labels_df=labels_df,
                        static=static_train, split=sp,
                        cfg=trainer_cfg_eff, task=cfg.task,
                        model_supports_static_branch=model_supports_static_branch,
                    )
                    test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                    test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])

                model_out = ensure_dir(out_root / "models" / model_name / sp.name)
                write_dl_artifacts(
                    out_dir=model_out, model=model, cfg=cfg,
                    feature_columns=feat_cols, code_vocab=None,
                )

            # --- Calibration ---
            cal_extra = {}
            if cfg.task.kind == "binary":
                if cfg.model.name in TABULAR_MODELS:
                    val_score = predict_tabular(art, X_val, cfg.task)
                    val_logits = predict_tabular_logits(art, X_val, cfg.task)
                    test_logits = predict_tabular_logits(art, X_test, cfg.task)
                    y_val_true = y_val.astype(float)

                y_score, cal_extra = maybe_calibrate_and_threshold(
                    cfg0=cfg0,
                    y_val_true=y_val_true.astype(float),
                    y_val_score=val_score.astype(float),
                    y_val_logits=val_logits,
                    y_test_score=y_score.astype(float),
                    y_test_logits=test_logits,
                )

            # --- Metrics ---
            finite_mask = np.isfinite(y_test.astype(float))
            y_test_eval = y_test[finite_mask]
            y_score_eval = np.asarray(y_score)[finite_mask]
            if y_test_eval.size == 0:
                continue
            if cfg.task.kind == "binary":
                metrics = binary_metrics(y_test_eval.astype(float), y_score_eval.astype(float)).metrics
            else:
                metrics = regression_metrics(y_test_eval.astype(float), y_score_eval.astype(float)).metrics
            metrics = {**metrics, **cal_extra}

            model_out = ensure_dir(out_root / "models" / model_name / sp.name)
            write_json(model_out / "metrics.json", metrics)

            run_records.append(
                {
                    "model": model_name,
                    "split": sp.name,
                    "task_kind": str(cfg.task.kind),
                    "prediction_mode": str(cfg.task.prediction_mode),
                    "skipped": 0,
                    "metrics": dict(metrics),
                    "artifacts": {
                        "metrics_json": str((model_out / "metrics.json").relative_to(out_root)),
                    },
                }
            )

            row = {
                "model": model_name,
                "split": sp.name,
                "skipped": 0,
                **metrics,
                "hpo_metric": cfg_model.hpo.metric,
                "hpo_mode": cfg_model.hpo.mode,
                "hpo_best_score": hpo_best_score,
                "hpo_best": hpo_best_str,
            }
            rows.append(row)

            if cfg.output.save_preds and len(test_patient_ids) > 0:
                preds = pd.DataFrame(
                    {"patient_id": test_patient_ids, "y_true": y_test, "y_pred": y_score}
                )
                if test_key is not None:
                    preds.insert(1, "bin_time", test_key["bin_time"].to_numpy())
                pred_dir = ensure_dir(out_root / "preds" / model_name)
                preds.to_parquet(pred_dir / f"{sp.name}.parquet", index=False)

    # --- Summary ---
    summary = pd.DataFrame(rows)

    write_json(
        out_root / "summary.json",
        {
            "run_name": str(cfg0.output.run_name),
            "task": {"kind": str(cfg0.task.kind), "prediction_mode": str(cfg0.task.prediction_mode)},
            "split": {"kind": str(cfg0.split.kind), "n_splits": int(cfg0.split.n_splits)},
            "calibration": {
                "enabled": bool(cfg0.calibration.enabled),
                "method": "temperature",
                "source": str(cfg0.calibration.source),
                "threshold_strategy": str(cfg0.calibration.threshold_strategy),
                "use_calibrated": bool(cfg0.calibration.use_calibrated),
            },
            "records": run_records,
        },
    )

    best_rows = [
        {"model": r.get("model"), "split": r["split"], "hpo_best": r.get("hpo_best")}
        for r in rows
    ]
    pd.DataFrame(best_rows).to_csv(out_root / "hpo_best.csv", index=False)


def _require_training_model(cfg):
    try:
        return cfg.require_model(context="training")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
