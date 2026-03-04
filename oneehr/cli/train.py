"""oneehr train subcommand."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from oneehr.eval.calibration import sigmoid
from oneehr.models.constants import TABULAR_MODELS, DL_MODELS, STATIC_ONLY_DL_MODELS
from oneehr.data.features import has_static_branch
from oneehr.cli._train_eval import maybe_calibrate_and_threshold, warn_unused_hpo_overrides
from oneehr.cli._train_dl import train_dl_patient_level, train_dl_time_level
from oneehr.utils.io import ensure_dir, write_json


def run_train(cfg_path: str, force: bool) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.artifacts.run_io import RunIO

    cfg0 = load_experiment_config(cfg_path)
    out_root = cfg0.output.root / cfg0.output.run_name

    if out_root.exists() and not force:
        raise SystemExit(
            f"Run directory already exists: {out_root}. "
            "Choose a new output.run_name or pass --force to overwrite."
        )
    ensure_dir(out_root)

    run = RunIO(run_root=out_root)
    _ = run.require_manifest()

    if force:
        import shutil

        for rel in ["models", "preds", "hpo", "splits", "summary.json", "hpo_best.csv"]:
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
    from oneehr.data.patient_index import make_patient_index, make_patient_index_from_static
    from oneehr.data.splits import Split, make_splits, save_splits, _parse_repeat_index
    from oneehr.data.postprocess import maybe_fit_transform_postprocess
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.hpo.grid import apply_overrides, iter_grid
    from oneehr.models.registry import build_model
    from oneehr.models.tabular import (
        predict_tabular,
        predict_tabular_logits,
        save_tabular_model,
        train_tabular_model,
    )
    from oneehr.modeling.persistence import write_dl_artifacts
    from oneehr.modeling.trainer import fit_model
    from oneehr.artifacts.run_io import RunIO
    from oneehr.cli._train_hpo import run_single_hpo, run_cv_mean_hpo, run_per_split_hpo

    cfg0 = load_experiment_config(cfg_path)
    train_dataset = cfg0.datasets.train if cfg0.datasets is not None else cfg0.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static_raw = load_static_table(train_dataset.static)
    if dynamic is not None:
        patient_index = make_patient_index(dynamic, "event_time", "patient_id")
    elif static_raw is not None:
        patient_index = make_patient_index_from_static(static_raw, patient_id_col="patient_id")
    else:
        raise SystemExit("dataset.dynamic or dataset.static is required for training.")
    splits = make_splits(patient_index, cfg0.split)

    out_root = cfg0.output.root / cfg0.output.run_name

    # Persist splits so that `oneehr test` can use them for self-split evaluation.
    save_splits(splits, out_root / "splits")

    # Expand splits with repeats for multi-seed training.
    if cfg0.trainer.repeat > 1:
        expanded: list[Split] = []
        for sp in splits:
            for r in range(cfg0.trainer.repeat):
                expanded.append(Split(
                    name=f"{sp.name}__r{r}",
                    train_patients=sp.train_patients,
                    val_patients=sp.val_patients,
                    test_patients=sp.test_patients,
                ))
        splits = expanded
        # Save expanded splits too (test command matches by name).
        save_splits(splits, out_root / "splits")
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

    models = cfg0.models or [cfg0.model]

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
            if static_only and cfg_model.task.prediction_mode != "patient":
                raise SystemExit("static-only datasets only support prediction_mode='patient'.")

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

            if static_only and model_name in DL_MODELS and model_name not in STATIC_ONLY_DL_MODELS:
                raise SystemExit(
                    f"Model {model_name!r} is a DL sequence model and requires dynamic features; "
                    "static-only datasets currently support ML models only."
                )

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
                if static_only:
                    if model_name != "mlp":
                        raise SystemExit("static-only DL training currently supports model.name='mlp' only.")
                    if static_train is None or static_feature_columns is None:
                        raise SystemExit("static-only DL training requires dataset.static and materialized static features.")
                    if cfg.task.prediction_mode != "patient":
                        raise SystemExit("static-only DL training supports prediction_mode='patient' only.")

                    pids_all = static_train.index.astype(str).to_numpy()
                    y_map = dict(zip(X.index.astype(str).tolist(), y.to_numpy().tolist()))
                    y_all = np.array([y_map.get(pid, np.nan) for pid in pids_all], dtype=np.float32)

                    tr_m = np.isin(pids_all, sp.train_patients) & np.isfinite(y_all)
                    va_m = np.isin(pids_all, sp.val_patients) & np.isfinite(y_all)
                    te_m = np.isin(pids_all, sp.test_patients) & np.isfinite(y_all)

                    if not bool(tr_m.any()) or not bool(va_m.any()) or not bool(te_m.any()):
                        raise SystemExit("No samples available after split for static-only DL training.")

                    X_static = torch.from_numpy(static_train.to_numpy(dtype=np.float32, copy=True))
                    L_static = torch.ones((X_static.shape[0],), dtype=torch.long)

                    X_tr, L_tr, y_tr = X_static[tr_m], L_static[tr_m], torch.from_numpy(y_all[tr_m])
                    X_va, L_va, y_va = X_static[va_m], L_static[va_m], torch.from_numpy(y_all[va_m])
                    X_te, L_te, y_te = X_static[te_m], L_static[te_m], torch.from_numpy(y_all[te_m])

                    input_dim = int(X_static.shape[1])
                    cfg_use = replace(cfg, _dynamic_dim=input_dim)
                    built = build_model(cfg_use)
                    model = built.model

                    trainer_cfg_eff = replace(cfg.trainer, seed=effective_seed)
                    fit = fit_model(
                        model=model,
                        X_train=X_tr, len_train=L_tr, y_train=y_tr, static_train=None,
                        X_val=X_va, len_val=L_va, y_val=y_va, static_val=None,
                        task=cfg.task, trainer=trainer_cfg_eff,
                    )

                    model.load_state_dict(fit.state_dict)
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(X_va, L_va).squeeze(-1).detach().cpu().numpy()
                        test_logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
                    if cfg.task.kind == "binary":
                        val_score = sigmoid(val_logits)
                        y_score = sigmoid(test_logits)
                    else:
                        val_score = val_logits
                        y_score = test_logits
                    y_val_true = y_va.detach().cpu().numpy()
                    y_test = y_te.detach().cpu().numpy()
                    test_patient_ids = pids_all[te_m]

                    model_out = ensure_dir(out_root / "models" / model_name / sp.name)
                    write_dl_artifacts(
                        out_dir=model_out, model=model, cfg=cfg,
                        feature_columns=list(static_feature_columns),
                        code_vocab=None,
                    )
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
                    if static_train is not None:
                        cfg_use = replace(cfg_use, _static_dim=int(static_train.shape[1]))
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
                "method": str(cfg0.calibration.method),
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

    if cfg0.split.kind.lower() == "time" and cfg0.split.inner_kind is not None:
        _run_final_prospective_eval(
            cfg0=cfg0, binned=binned, labels_df=labels_df,
            patient_index=patient_index, out_root=out_root,
        )


def _run_final_prospective_eval(
    *,
    cfg0,
    binned,
    labels_df,
    patient_index,
    out_root: Path,
) -> None:
    import pandas as pd

    from oneehr.config.load import load_experiment_config
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.eval.bootstrap import bootstrap_metric
    from oneehr.hpo.grid import apply_overrides
    from oneehr.models.tabular import predict_tabular, train_tabular_model
    from oneehr.artifacts.run_io import RunIO

    run = RunIO(run_root=out_root)
    manifest = run.require_manifest()

    if cfg0.trainer.final_model_source not in {"refit", "best_split"}:
        raise SystemExit("trainer.final_model_source must be 'refit' or 'best_split'")

    boundary = pd.to_datetime(cfg0.split.time_boundary, errors="raise")
    pid = patient_index[["patient_id", "max_time"]].copy()
    pid["patient_id"] = pid["patient_id"].astype(str)
    pre_patients = pid[pid["max_time"] < boundary]["patient_id"].to_numpy().astype(str)
    post_patients = pid[pid["max_time"] >= boundary]["patient_id"].to_numpy().astype(str)

    if cfg0.task.prediction_mode != "patient":
        raise SystemExit("final prospective eval currently supports prediction_mode='patient' only")

    X, y = run.load_patient_view(manifest)

    pre_mask = X.index.astype(str).isin(pre_patients)
    post_mask = X.index.astype(str).isin(post_patients)

    if cfg0.trainer.final_refit not in {"train_only", "train_val"}:
        raise SystemExit("trainer.final_refit must be 'train_only' or 'train_val'")

    if cfg0.trainer.final_refit == "train_only" and cfg0.split.val_size > 0:
        rng = np.random.default_rng(cfg0.split.seed)
        pre_idx = np.where(pre_mask)[0]
        n_val = max(1, int(round(len(pre_idx) * cfg0.split.val_size)))
        perm = rng.permutation(len(pre_idx))
        val_idx = pre_idx[perm[:n_val]]
        train_idx = pre_idx[perm[n_val:]]
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[train_idx] = True
    else:
        train_mask = pre_mask

    X_train, y_train = X.loc[train_mask], y.loc[train_mask].to_numpy()
    X_test, y_test = X.loc[post_mask], y.loc[post_mask].to_numpy()

    final_dir = ensure_dir(out_root / "final")

    best_split_by_model: dict[str, str] = {}
    if cfg0.trainer.final_model_source == "best_split":
        summary_path = out_root / "summary.json"
        if not summary_path.exists():
            raise SystemExit("Missing summary.json for selecting best_split model.")
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        recs = summary_payload.get("records") or []
        if not isinstance(recs, list):
            raise SystemExit("Invalid summary.json format: records must be a list")
        summary = pd.DataFrame(
            [
                {
                    "model": r.get("model"),
                    "split": r.get("split"),
                    "skipped": r.get("skipped", 0),
                    **(r.get("metrics") or {}),
                }
                for r in recs
            ]
        )
        metric = cfg0.hpo.metric
        if metric in {"val_auroc", "auroc"}:
            metric = "auroc"
        elif metric in {"val_auprc", "auprc"}:
            metric = "auprc"
        elif metric in {"val_rmse", "rmse"}:
            metric = "rmse"
        elif metric in {"val_mae", "mae"}:
            metric = "mae"
        if metric not in summary.columns:
            metric = "auroc" if cfg0.task.kind == "binary" else "rmse"
        for m_name, dfm in summary.groupby("model"):
            dfm2 = dfm[dfm.get("skipped", 0) == 0].copy()
            if dfm2.empty or metric not in dfm2.columns:
                continue
            if cfg0.task.kind == "binary":
                best_row = dfm2.sort_values(metric, ascending=False).iloc[0]
            else:
                best_row = dfm2.sort_values(metric, ascending=True).iloc[0]
            best_split_by_model[str(m_name)] = str(best_row["split"])

    rows = []
    for model_cfg in (cfg0.models or [cfg0.model]):
        cfg_model = replace(cfg0, model=model_cfg, models=[model_cfg])
        model_name = cfg_model.model.name
        if model_name in cfg0.hpo_by_model:
            cfg_model = replace(cfg_model, hpo=cfg0.hpo_by_model[model_name])

        if cfg_model.model.name != "xgboost":
            continue

        if cfg0.trainer.final_model_source == "best_split":
            split_name = best_split_by_model.get(model_name)
            if split_name is None:
                continue
            model_dir = out_root / "models" / model_name / split_name
            model_path = model_dir / "model.json"
            cols_path = model_dir / "feature_columns.json"
            if not model_path.exists() or not cols_path.exists():
                raise SystemExit(f"Missing saved model artifacts for best_split at {model_dir}")
            feature_columns = json.loads(cols_path.read_text(encoding="utf-8"))
            from xgboost import XGBClassifier, XGBRegressor

            if cfg0.task.kind == "binary":
                mdl = XGBClassifier()
                mdl.load_model(model_path)
                y_pred = mdl.predict_proba(X_test[feature_columns])[:, 1]
            else:
                mdl = XGBRegressor()
                mdl.load_model(model_path)
                y_pred = mdl.predict(X_test[feature_columns])
            cfg_fit = cfg_model
        else:
            hpo_dir = out_root / "hpo" / model_name
            overrides = {}
            if cfg_model.hpo.enabled:
                if cfg_model.hpo.scope == "cv_mean":
                    sel = hpo_dir / "select_best_split.json"
                    if sel.exists():
                        overrides = (
                            json.loads(sel.read_text(encoding="utf-8")).get("selected_overrides") or {}
                        )
                elif cfg_model.hpo.scope == "single":
                    sel = hpo_dir / "best_once.json"
                    if sel.exists():
                        best = json.loads(sel.read_text(encoding="utf-8")).get("best") or {}
                        overrides = best.get("overrides") or {}
            cfg_fit = apply_overrides(cfg_model, overrides)

            if cfg_fit.task.kind == "binary" and len(np.unique(y_train)) < 2:
                continue
            art = train_tabular_model(
                model_name="xgboost",
                X_train=X_train, y_train=y_train,
                X_val=None, y_val=None,
                task=cfg_fit.task, model_cfg=cfg_fit.model.xgboost,
            )
            y_pred = predict_tabular(art, X_test, cfg_fit.task)

        if cfg_fit.task.kind == "binary":
            metrics = binary_metrics(y_test.astype(float), y_pred.astype(float)).metrics
        else:
            metrics = regression_metrics(y_test.astype(float), y_pred.astype(float)).metrics

        write_json(final_dir / f"test_metrics_{model_name}.json", metrics)

        if cfg_fit.trainer.bootstrap_test:
            metric_name = "auroc" if cfg_fit.task.kind == "binary" else "rmse"
            bs = bootstrap_metric(
                y_true=y_test.astype(float), y_pred=y_pred.astype(float),
                task=cfg_fit.task, metric=metric_name,
                n=cfg_fit.trainer.bootstrap_n, seed=cfg_fit.trainer.seed,
            )
            write_json(
                final_dir / f"test_bootstrap_{model_name}.json",
                {
                    "metric": bs.metric, "n": bs.n,
                    "mean": bs.mean, "ci_low": bs.ci_low, "ci_high": bs.ci_high,
                },
            )

        if cfg_fit.output.save_preds and len(X_test) > 0:
            pd.DataFrame(
                {"patient_id": X_test.index.astype(str), "y_true": y_test, "y_pred": y_pred}
            ).to_parquet(final_dir / f"test_preds_{model_name}.parquet", index=False)

        rows.append({"model": model_name, **metrics})

    if rows:
        write_json(final_dir / "test_summary.json", {"records": rows})
