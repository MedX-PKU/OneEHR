"""oneehr train subcommand."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from oneehr.eval.calibration import sigmoid
from oneehr.utils.io import ensure_dir, write_json


def _train_sequence_patient_level(
    model,
    binned,
    y,
    static,
    split,
    cfg,
    task,
    model_supports_static_branch: bool = False,
    *,
    y_map: dict[str, float] | None = None,
):
    from oneehr.data.sequence import build_patient_sequences, pad_sequences

    feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    pids, seqs, lens = build_patient_sequences(binned, feat_cols)
    X_seq = pad_sequences(seqs, lens)
    lens_t = torch.from_numpy(lens)

    if y_map is None:
        y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
    y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)
    pids_arr = np.array(pids, dtype=str)

    train_m = np.isin(pids_arr, split.train_patients)
    val_m = np.isin(pids_arr, split.val_patients)
    test_m = np.isin(pids_arr, split.test_patients)

    finite_y = np.isfinite(y_arr)
    train_m = train_m & finite_y
    val_m = val_m & finite_y
    test_m = test_m & finite_y

    if not bool(train_m.any()) or not bool(val_m.any()) or not bool(test_m.any()):
        raise SystemExit(
            "No samples available for DL sequence training in this split. "
            "Check split configuration (train/val/test sizes)."
        )

    X_tr, L_tr, y_tr = X_seq[train_m], lens_t[train_m], torch.from_numpy(y_arr[train_m])
    X_va, L_va, y_va = X_seq[val_m], lens_t[val_m], torch.from_numpy(y_arr[val_m])
    X_te, L_te, y_te = X_seq[test_m], lens_t[test_m], torch.from_numpy(y_arr[test_m])

    S = None
    if static is not None and model_supports_static_branch:
        from oneehr.data.sequence import align_static_features

        S_all = align_static_features(
            pids,
            static,
            expected_feature_columns=list(static.columns),
        )
        if S_all is not None:
            S = torch.from_numpy(np.asarray(S_all, dtype=np.float32).copy())
            S_tr, S_va, S_te = S[train_m], S[val_m], S[test_m]
        else:
            S_tr = S_va = S_te = None
    else:
        S_tr = S_va = S_te = None

    if static is not None and not model_supports_static_branch:
        # Repeat patient-level static covariates across time and concat to dynamic bins.
        from oneehr.data.sequence import align_static_features

        S_all = align_static_features(
            pids,
            static,
            expected_feature_columns=list(static.columns),
        )
        if S_all is not None:
            S_np = np.asarray(S_all, dtype=np.float32)
            S_rep = np.repeat(S_np[:, None, :], X_seq.shape[1], axis=1)
            X_seq = torch.from_numpy(np.concatenate([X_seq, S_rep], axis=-1).astype(np.float32, copy=False))
            X_tr, X_va, X_te = X_seq[train_m], X_seq[val_m], X_seq[test_m]
        S_tr = S_va = S_te = None

    from oneehr.modeling.trainer import fit_sequence_model

    fit = fit_sequence_model(
        model=model,
        X_train=X_tr,
        len_train=L_tr,
        y_train=y_tr,
        static_train=S_tr,
        X_val=X_va,
        len_val=L_va,
        y_val=y_va,
        static_val=S_va,
        task=task,
        trainer=cfg,
    )

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_va is None:
            val_logits = model(X_va, L_va).squeeze(-1).detach().cpu().numpy()
        else:
            val_logits = model(X_va, L_va, S_va).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        val_score = sigmoid(val_logits)
    else:
        val_score = val_logits

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_te is None:
            logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
        else:
            logits = model(X_te, L_te, S_te).squeeze(-1).detach().cpu().numpy()
    y_true = y_te.detach().cpu().numpy()
    if task.kind == "binary":
        y_score = sigmoid(logits)
        return y_score, y_true, pids_arr[test_m], logits, val_score, val_logits, y_va.detach().cpu().numpy()
    y_score = logits
    return y_score, y_true, pids_arr[test_m], None, val_score, val_logits, y_va.detach().cpu().numpy()


def _train_sequence_time_level(
    model,
    binned,
    labels_df,
    static,
    split,
    cfg,
    task,
    model_supports_static_branch: bool = False,
):
    from oneehr.data.sequence import build_time_sequences, pad_sequences

    feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
        binned,
        labels_df,
        feat_cols,
        label_time_col="bin_time",
    )
    X_seq = pad_sequences(seqs, lens)
    Y_seq = pad_sequences([y[:, None] for y in y_seqs], lens).squeeze(-1)
    M_seq = pad_sequences([m[:, None] for m in mask_seqs], lens).squeeze(-1)
    lens_t = torch.from_numpy(lens)

    pids_arr = np.array(pids, dtype=str)
    train_m = np.isin(pids_arr, split.train_patients)
    val_m = np.isin(pids_arr, split.val_patients)
    test_m = np.isin(pids_arr, split.test_patients)

    X_tr, L_tr, Y_tr, M_tr = (
        X_seq[train_m],
        lens_t[train_m],
        Y_seq[train_m],
        M_seq[train_m],
    )
    X_va, L_va, Y_va, M_va = (
        X_seq[val_m],
        lens_t[val_m],
        Y_seq[val_m],
        M_seq[val_m],
    )
    X_te, L_te, Y_te, M_te = (
        X_seq[test_m],
        lens_t[test_m],
        Y_seq[test_m],
        M_seq[test_m],
    )

    if static is not None and model_supports_static_branch:
        from oneehr.data.sequence import align_static_features

        S_all = align_static_features(pids, static, expected_feature_columns=list(static.columns))
        if S_all is not None:
            S = torch.from_numpy(np.asarray(S_all, dtype=np.float32).copy())
            S_tr, S_va, S_te = S[train_m], S[val_m], S[test_m]
        else:
            S_tr = S_va = S_te = None
    else:
        S_tr = S_va = S_te = None

    if static is not None and not model_supports_static_branch:
        from oneehr.data.sequence import align_static_features

        S_all = align_static_features(pids, static, expected_feature_columns=list(static.columns))
        if S_all is not None:
            S_np = np.asarray(S_all, dtype=np.float32)
            S_rep = np.repeat(S_np[:, None, :], X_seq.shape[1], axis=1)
            X_seq = torch.from_numpy(np.concatenate([X_seq, S_rep], axis=-1).astype(np.float32, copy=False))
            X_tr, X_va, X_te = X_seq[train_m], X_seq[val_m], X_seq[test_m]
        S_tr = S_va = S_te = None

    from oneehr.modeling.trainer import fit_sequence_model_time

    fit = fit_sequence_model_time(
        model=model,
        X_train=X_tr,
        len_train=L_tr,
        y_train=Y_tr,
        mask_train=M_tr,
        static_train=S_tr,
        X_val=X_va,
        len_val=L_va,
        y_val=Y_va,
        mask_val=M_va,
        static_val=S_va,
        task=task,
        trainer=cfg,
    )

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_va is None:
            val_logits = model(X_va, L_va).squeeze(-1).detach().cpu().numpy()
        else:
            val_logits = model(X_va, L_va, S_va).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        val_score = sigmoid(val_logits)
    else:
        val_score = val_logits
    y_val_np = Y_va.detach().cpu().numpy()
    m_val_np = M_va.detach().cpu().numpy() if M_va is not None else None
    if m_val_np is not None:
        flat = m_val_np.reshape(-1).astype(bool)
        val_score = val_score.reshape(-1)[flat]
        val_logits = val_logits.reshape(-1)[flat]
        y_val_np = y_val_np.reshape(-1)[flat]

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_te is None:
            logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
        else:
            logits = model(X_te, L_te, S_te).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        y_score = sigmoid(logits)
    else:
        y_score = logits
    y_true = Y_te.detach().cpu().numpy()
    mask = M_te.detach().cpu().numpy()
    if mask is not None:
        flat = mask.reshape(-1).astype(bool)
        y_score = y_score.reshape(-1)[flat]
        y_true = y_true.reshape(-1)[flat]
        logits = logits.reshape(-1)[flat]

    key_rows = []
    for pid, t, m in zip(pids, time_seqs, mask_seqs, strict=True):
        for tt, mm in zip(t, m, strict=True):
            if bool(mm):
                key_rows.append((str(pid), tt))

    if task.kind == "binary":
        return y_score, y_true, key_rows, logits, val_score, val_logits, y_val_np
    return y_score, y_true, key_rows, None, val_score, val_logits, y_val_np


def _maybe_calibrate_and_threshold(
    *,
    cfg0,
    y_val_true: np.ndarray,
    y_val_score: np.ndarray,
    y_val_logits: np.ndarray | None,
    y_test_score: np.ndarray,
    y_test_logits: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply optional calibration using val split, and compute thresholds."""

    from oneehr.eval.calibration import (
        binary_brier,
        binary_log_loss,
        calibrate_from_logits,
        calibrate_from_probs,
        select_threshold_f1,
    )

    extra: dict[str, float] = {}
    if cfg0.task.kind != "binary" or not cfg0.calibration.enabled:
        return y_test_score, extra

    if cfg0.calibration.source != "val":
        raise SystemExit("calibration.source currently supports 'val' only")
    if cfg0.calibration.threshold_strategy != "f1":
        raise SystemExit("calibration.threshold_strategy currently supports 'f1' only")

    method = cfg0.calibration.method
    y_val_true = y_val_true.astype(float).reshape(-1)
    y_val_score = y_val_score.astype(float).reshape(-1)

    if y_val_logits is not None:
        y_val_cal, params = calibrate_from_logits(y_val_true, y_val_logits, method=method)
    else:
        y_val_cal, params = calibrate_from_probs(y_val_true, y_val_score, method=method)

    thr_raw = select_threshold_f1(y_val_true, y_val_score)
    thr_cal = select_threshold_f1(y_val_true, y_val_cal)
    extra["val_best_threshold_raw_f1"] = float(thr_raw)
    extra["val_best_threshold_cal_f1"] = float(thr_cal)

    extra["val_logloss_raw"] = binary_log_loss(y_val_true, y_val_score)
    extra["val_brier_raw"] = binary_brier(y_val_true, y_val_score)
    extra["val_logloss_cal"] = binary_log_loss(y_val_true, y_val_cal)
    extra["val_brier_cal"] = binary_brier(y_val_true, y_val_cal)

    for k, v in params.items():
        extra[f"calibration_{k}"] = float(v)

    if not cfg0.calibration.use_calibrated:
        return y_test_score, extra

    if method == "temperature":
        t = float(params["temperature"])
        if y_test_logits is not None:
            y_test_cal = sigmoid(y_test_logits / t)
        else:
            eps = np.finfo(float).eps
            p = np.clip(y_test_score.astype(float), eps, 1.0 - eps)
            raw_logits = np.log(p / (1.0 - p))
            y_test_cal = sigmoid(raw_logits / t)
        return y_test_cal.astype(float), extra

    if method == "platt":
        a = float(params["a"])
        b = float(params["b"])
        if y_test_logits is not None:
            z = a * y_test_logits + b
        else:
            eps = np.finfo(float).eps
            p = np.clip(y_test_score.astype(float), eps, 1.0 - eps)
            raw_logits = np.log(p / (1.0 - p))
            z = a * raw_logits + b
        y_test_cal = sigmoid(z)
        return y_test_cal.astype(float), extra

    raise SystemExit(f"Unsupported calibration.method={method!r}")


def _warn_unused_hpo_overrides(model_name: str, overrides: list[dict[str, object]]) -> None:
    valid_model_key = f"model.{model_name}."
    invalid_model_keys: set[str] = set()
    trainer_keys: set[str] = set()
    for override in overrides:
        for key in override.keys():
            if key.startswith("model.") and not key.startswith(valid_model_key):
                invalid_model_keys.add(key)
            if model_name == "xgboost" and key.startswith("trainer."):
                trainer_keys.add(key)
    if invalid_model_keys:
        print(
            "HPO overrides include keys not matching model "
            f"{model_name!r}: {sorted(invalid_model_keys)}",
            file=sys.stderr,
        )
    if trainer_keys:
        print(
            "HPO overrides include trainer.* keys, which are ignored for xgboost: "
            f"{sorted(trainer_keys)}",
            file=sys.stderr,
        )


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

        for rel in ["models", "preds", "hpo", "summary.json", "hpo_best.csv"]:
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
    from oneehr.data.sequence import build_patient_sequences
    from oneehr.data.splits import make_splits
    from oneehr.data.postprocess import maybe_fit_transform_postprocess
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.hpo.grid import apply_overrides, iter_grid
    from oneehr.hpo.runner import select_best_with_trials
    from oneehr.models.registry import build_model
    from oneehr.models.tabular import (
        predict_tabular,
        predict_tabular_logits,
        save_tabular_model,
        train_tabular_model,
    )
    from oneehr.modeling.persistence import write_dl_artifacts
    from oneehr.modeling.trainer import fit_sequence_model, fit_sequence_model_time
    from oneehr.artifacts.run_io import RunIO
    from oneehr.eval.bootstrap import bootstrap_metric

    cfg0 = load_experiment_config(cfg_path)
    train_dataset = cfg0.datasets.train if cfg0.datasets is not None else cfg0.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static_raw = load_static_table(train_dataset.static)
    if dynamic is not None:
        patient_index = make_patient_index(
            dynamic,
            "event_time",
            "patient_id",
        )
    elif static_raw is not None:
        patient_index = make_patient_index_from_static(static_raw, patient_id_col="patient_id")
    else:
        raise SystemExit("dataset.dynamic or dataset.static is required for training.")
    splits = make_splits(patient_index, cfg0.split)

    out_root = cfg0.output.root / cfg0.output.run_name
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

    TABULAR_MODELS = {"xgboost", "catboost", "rf", "dt", "gbdt"}
    DL_MODELS = {"gru", "rnn", "lstm", "mlp", "tcn", "transformer", "adacare", "stagenet", "retain", "concare", "grasp", "mcgru", "dragent"}
    STATIC_ONLY_DL_MODELS = {"mlp"}

    models = cfg0.models or [cfg0.model]

    rows = []
    run_records: list[dict[str, object]] = []

    for model_cfg in models:
        cfg_model = replace(cfg0, model=model_cfg, models=[model_cfg])
        model_name = cfg_model.model.name
        if model_name in cfg0.hpo_by_model:
            cfg_model = replace(cfg_model, hpo=cfg0.hpo_by_model[model_name])

        _warn_unused_hpo_overrides(model_name, list(iter_grid(cfg_model.hpo)))

        reused_overrides: dict[str, object] | None = None
        if cfg_model.hpo.enabled and cfg_model.hpo.scope in {"single", "cv_mean"}:
            if cfg_model.hpo.tune_split is not None:
                tune_splits = [sp for sp in splits if sp.name == cfg_model.hpo.tune_split]
                if not tune_splits:
                    raise SystemExit(
                        f"hpo.tune_split={cfg_model.hpo.tune_split!r} not found in splits: "
                        f"{[sp.name for sp in splits]}"
                    )
                tune_split = tune_splits[0]
            else:
                tune_split = splits[0]

            def _make_masks(sp):
                if cfg_model.task.prediction_mode == "patient":
                    tr = X.index.astype(str).isin(sp.train_patients)
                    va = X.index.astype(str).isin(sp.val_patients)
                else:
                    assert key is not None
                    tr = key["patient_id"].astype(str).isin(sp.train_patients)
                    va = key["patient_id"].astype(str).isin(sp.val_patients)
                if global_mask is not None:
                    tr = tr & global_mask
                    va = va & global_mask
                return tr, va

            tr_m, va_m = _make_masks(tune_split)
            X_train_h, y_train_h = X.loc[tr_m], y.loc[tr_m].to_numpy()
            X_val_h, y_val_h = X.loc[va_m], y.loc[va_m].to_numpy()

            def _eval_trial_once(cfg) -> tuple[float, dict[str, float]] | None:
                hpo_metric = cfg.hpo.metric
                if cfg.model.name != "xgboost":
                    return None
                if cfg.task.kind == "binary" and len(np.unique(y_train_h)) < 2:
                    return None
                art = train_tabular_model(
                    model_name="xgboost",
                    X_train=X_train_h,
                    y_train=y_train_h,
                    X_val=X_val_h,
                    y_val=y_val_h,
                    task=cfg.task,
                    model_cfg=cfg.model.xgboost,
                )
                y_val_score = predict_tabular(art, X_val_h, cfg.task)
                if cfg.task.kind == "binary":
                    vm = binary_metrics(y_val_h.astype(float), y_val_score.astype(float)).metrics
                    if hpo_metric in {"val_auroc", "auroc"}:
                        return float(vm["auroc"]), vm
                    return float(vm["auprc"]), vm
                vm = regression_metrics(y_val_h.astype(float), y_val_score.astype(float)).metrics
                if hpo_metric in {"val_rmse", "rmse"}:
                    return float(vm["rmse"]), vm
                return float(vm["mae"]), vm

            hpo_res_once = select_best_with_trials(cfg_model, _eval_trial_once)
            reused_overrides = hpo_res_once.best.overrides if hpo_res_once.best is not None else {}
            hpo_dir = ensure_dir(out_root / "hpo" / model_name)
            write_json(
                hpo_dir / "best_once.json",
                {
                    "split": tune_split.name,
                    "metric": cfg_model.hpo.metric,
                    "mode": cfg_model.hpo.mode,
                    "scope": cfg_model.hpo.scope,
                    "best": None if hpo_res_once.best is None else {
                        "score": hpo_res_once.best.score,
                        "overrides": hpo_res_once.best.overrides,
                        "metrics": hpo_res_once.best.metrics,
                    },
                },
            )

        if cfg_model.hpo.enabled and cfg_model.hpo.scope == "cv_mean":
            trials = []
            best = None

            def _better(a: float, b: float) -> bool:
                return a < b if cfg_model.hpo.mode == "min" else a > b

            for overrides in iter_grid(cfg_model.hpo):
                cfg_trial = apply_overrides(cfg_model, overrides)
                split_scores = []
                split_metrics = []

                for sp in splits:
                    if cfg_model.task.prediction_mode == "patient":
                        train_mask = X.index.astype(str).isin(sp.train_patients)
                        val_mask = X.index.astype(str).isin(sp.val_patients)
                    else:
                        assert key is not None
                        train_mask = key["patient_id"].astype(str).isin(sp.train_patients)
                        val_mask = key["patient_id"].astype(str).isin(sp.val_patients)
                    if global_mask is not None:
                        train_mask = train_mask & global_mask
                        val_mask = val_mask & global_mask

                    X_train_s, y_train_s = X.loc[train_mask], y.loc[train_mask].to_numpy()
                    X_val_s, y_val_s = X.loc[val_mask], y.loc[val_mask].to_numpy()

                    if cfg_trial.model.name != "xgboost":
                        continue
                    if cfg_trial.task.kind == "binary" and len(np.unique(y_train_s)) < 2:
                        continue
                    art = train_tabular_model(
                        model_name="xgboost",
                        X_train=X_train_s,
                        y_train=y_train_s,
                        X_val=X_val_s,
                        y_val=y_val_s,
                        task=cfg_trial.task,
                        model_cfg=cfg_trial.model.xgboost,
                    )
                    y_val_score = predict_tabular(art, X_val_s, cfg_trial.task)

                    if cfg_trial.task.kind == "binary":
                        vm = binary_metrics(y_val_s.astype(float), y_val_score.astype(float)).metrics
                        if cfg_trial.hpo.aggregate_metric is not None:
                            if cfg_trial.hpo.aggregate_metric not in vm:
                                raise SystemExit(
                                    f"hpo.aggregate_metric={cfg_trial.hpo.aggregate_metric!r} not in metrics: {list(vm.keys())}"
                                )
                            score = float(vm[cfg_trial.hpo.aggregate_metric])
                        else:
                            score = float(vm["auroc"]) if cfg_trial.hpo.metric in {"val_auroc", "auroc"} else float(vm["auprc"])
                    else:
                        vm = regression_metrics(y_val_s.astype(float), y_val_score.astype(float)).metrics
                        if cfg_trial.hpo.aggregate_metric is not None:
                            if cfg_trial.hpo.aggregate_metric not in vm:
                                raise SystemExit(
                                    f"hpo.aggregate_metric={cfg_trial.hpo.aggregate_metric!r} not in metrics: {list(vm.keys())}"
                                )
                            score = float(vm[cfg_trial.hpo.aggregate_metric])
                        else:
                            score = float(vm["rmse"]) if cfg_trial.hpo.metric in {"val_rmse", "rmse"} else float(vm["mae"])

                    split_scores.append({"split": sp.name, "score": score})
                    split_metrics.append(vm)

                if not split_scores:
                    continue

                mean_score = float(np.mean([r["score"] for r in split_scores]))
                trial = {"mean_score": mean_score, "overrides": dict(overrides), "per_split": split_scores}
                trials.append(trial)

                if best is None or _better(mean_score, best["mean_score"]):
                    best = trial

            hpo_dir = ensure_dir(out_root / "hpo" / model_name)
            write_json(
                hpo_dir / "select_best_split.json",
                {
                    "metric": cfg_model.hpo.metric,
                    "mode": cfg_model.hpo.mode,
                    "selected_mean_score": None if best is None else best["mean_score"],
                    "selected_overrides": None if best is None else best["overrides"],
                    "trials": trials,
                },
            )
            reused_overrides = {} if best is None else best["overrides"]

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
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                pipeline=cfg0.preprocess.pipeline,
            )

            static_train = None
            if static_all is not None and static_feature_columns is not None:
                static_train = static_all
            if static_all is not None and static_feature_columns is not None and model_name in TABULAR_MODELS:
                # ML models always use concat (per pyehr pipeline): join static into X.
                # Avoid collisions when a feature exists in both dynamic and static spaces.
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

            def _eval_trial(cfg) -> tuple[float, dict[str, float]] | None:
                hpo_metric = cfg.hpo.metric

                if cfg.model.name in TABULAR_MODELS:
                    if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                        return None
                    tab_model_cfg = getattr(cfg.model, cfg.model.name)
                    art = train_tabular_model(
                        model_name=cfg.model.name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        task=cfg.task,
                        model_cfg=tab_model_cfg,
                    )
                    y_val_score = predict_tabular(art, X_val, cfg.task)
                    if cfg.task.kind == "binary":
                        vm = binary_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
                        if hpo_metric in {"val_auroc", "auroc"}:
                            return float(vm["auroc"]), vm
                        return float(vm["auprc"]), vm
                    vm = regression_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
                    if hpo_metric in {"val_rmse", "rmse"}:
                        return float(vm["rmse"]), vm
                    return float(vm["mae"]), vm

                if len(dynamic_feature_columns) == 0:
                    return None

                from oneehr.data.sequence import build_patient_sequences, build_time_sequences, pad_sequences

                feat_cols = list(dynamic_feature_columns)

                if cfg.task.prediction_mode == "patient":
                    pids, seqs, lens = build_patient_sequences(binned, feat_cols)
                    X_seq = pad_sequences(seqs, lens)
                    lens_t = torch.from_numpy(lens)

                    y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
                    y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)
                    pids_arr = np.array(pids, dtype=str)

                    tr_m = np.isin(pids_arr, sp.train_patients)
                    va_m = np.isin(pids_arr, sp.val_patients)
                    X_tr, L_tr, y_tr = (
                        X_seq[tr_m],
                        lens_t[tr_m],
                        torch.from_numpy(y_arr[tr_m]),
                    )
                    X_va, L_va, y_va = (
                        X_seq[va_m],
                        lens_t[va_m],
                        torch.from_numpy(y_arr[va_m]),
                    )

                    input_dim = int(X_seq.shape[-1])
                    cfg_use = replace(cfg, _dynamic_dim=input_dim)
                    built = build_model(cfg_use)
                    if built.kind != "dl":
                        return None
                    mdl = built.model

                    fit = fit_sequence_model(
                        model=mdl,
                        X_train=X_tr,
                        len_train=L_tr,
                        y_train=y_tr,
                        static_train=None,
                        X_val=X_va,
                        len_val=L_va,
                        y_val=y_va,
                        static_val=None,
                        task=cfg.task,
                        trainer=cfg.trainer,
                    )
                    last = fit.history[-1] if fit.history else {}
                    monitor = cfg.trainer.monitor
                    score = float(last.get(monitor, last.get("val_loss", 0.0)))
                    return score, {monitor: score}

                if cfg.task.prediction_mode == "time":
                    if labels_df is None:
                        raise SystemExit("prediction_mode='time' requires labels (labels.fn).")

                    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
                        binned,
                        labels_df,
                        feat_cols,
                        label_time_col="bin_time",
                    )
                    X_seq = pad_sequences(seqs, lens)
                    Y_seq = pad_sequences([yy[:, None] for yy in y_seqs], lens).squeeze(-1)
                    M_seq = pad_sequences([mm[:, None] for mm in mask_seqs], lens).squeeze(-1)
                    lens_t = torch.from_numpy(lens)

                    pids_arr = np.array(pids, dtype=str)
                    tr_m = np.isin(pids_arr, sp.train_patients)
                    va_m = np.isin(pids_arr, sp.val_patients)
                    X_tr, L_tr, Y_tr, M_tr = (
                        X_seq[tr_m],
                        lens_t[tr_m],
                        Y_seq[tr_m],
                        M_seq[tr_m],
                    )
                    X_va, L_va, Y_va, M_va = (
                        X_seq[va_m],
                        lens_t[va_m],
                        Y_seq[va_m],
                        M_seq[va_m],
                    )

                    input_dim = int(X_seq.shape[-1])
                    cfg_use = replace(cfg, _dynamic_dim=input_dim)
                    built = build_model(cfg_use)
                    if built.kind != "dl":
                        return None
                    mdl = built.model

                    fit = fit_sequence_model_time(
                        model=mdl,
                        X_train=X_tr,
                        len_train=L_tr,
                        y_train=Y_tr,
                        mask_train=M_tr,
                        static_train=None,
                        X_val=X_va,
                        len_val=L_va,
                        y_val=Y_va,
                        mask_val=M_va,
                        static_val=None,
                        task=cfg.task,
                        trainer=cfg.trainer,
                    )
                    last = fit.history[-1] if fit.history else {}
                    monitor = cfg.trainer.monitor
                    score = float(last.get(monitor, last.get("val_loss", 0.0)))
                    return score, {monitor: score}

                return None

            if cfg_model.hpo.enabled and cfg_model.hpo.scope == "per_split":
                hpo_res = select_best_with_trials(cfg_model, _eval_trial)
                best_overrides = hpo_res.best.overrides if hpo_res.best is not None else {}

                trial_rows = []
                for tr in hpo_res.trials:
                    trial_rows.append(
                        {
                            "model": model_name,
                            "split": sp.name,
                            "hpo_metric": cfg_model.hpo.metric,
                            "hpo_mode": cfg_model.hpo.mode,
                            "trial_score": tr.score,
                            "overrides": str(tr.overrides),
                            **{f"trial_{k}": v for k, v in tr.metrics.items()},
                        }
                    )

                hpo_dir = ensure_dir(out_root / "hpo" / model_name)
                pd.DataFrame(trial_rows).to_csv(hpo_dir / f"trials_{sp.name}.csv", index=False)
                write_json(
                    hpo_dir / f"best_{sp.name}.json",
                    {
                        "split": sp.name,
                        "metric": cfg_model.hpo.metric,
                        "mode": cfg_model.hpo.mode,
                        "best": None if hpo_res.best is None else {
                            "score": hpo_res.best.score,
                            "overrides": hpo_res.best.overrides,
                            "metrics": hpo_res.best.metrics,
                        },
                    },
                )
                cfg = apply_overrides(cfg_model, best_overrides)
                hpo_best_score = None if hpo_res.best is None else hpo_res.best.score
                hpo_best_str = str(best_overrides)
            else:
                if reused_overrides is not None:
                    cfg = apply_overrides(cfg_model, reused_overrides)
                    hpo_best_str = str(reused_overrides)
                else:
                    cfg = cfg_model
                    hpo_best_str = ""
                hpo_best_score = None

            # Train on this split using the selected hyperparameters, then evaluate on test.
            if cfg.model.name in TABULAR_MODELS:
                if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                    row = {
                        "model": model_name,
                        "split": sp.name,
                        "skipped": 1,
                        "reason": "single_class_train",
                    }
                    rows.append(row)
                    continue

                tab_model_cfg = getattr(cfg.model, cfg.model.name)
                art = train_tabular_model(
                    model_name=cfg.model.name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    task=cfg.task,
                    model_cfg=tab_model_cfg,
                )
                model_out = ensure_dir(out_root / "models" / model_name / sp.name)
                save_tabular_model(art, model_out)
                y_score = predict_tabular(art, X_test, cfg.task)
                if fitted_post is not None:
                    pp_dir = ensure_dir(out_root / "preprocess" / sp.name)
                    write_json(pp_dir / "pipeline.json", {"pipeline": fitted_post.pipeline})
            else:
                if static_only:
                    # Static-only DL (tabular) currently supported for MLP only.
                    if model_name != "mlp":
                        raise SystemExit("static-only DL training currently supports model.name='mlp' only.")
                    if static_train is None or static_feature_columns is None:
                        raise SystemExit("static-only DL training requires dataset.static and materialized static features.")
                    if cfg.task.prediction_mode != "patient":
                        raise SystemExit("static-only DL training supports prediction_mode='patient' only.")

                    # Build per-patient tabular tensors from static features.
                    # Align X/y by patient_id and apply the same split masks.
                    pids_all = static_train.index.astype(str).to_numpy()
                    y_map = dict(zip(X.index.astype(str).tolist(), y.to_numpy().tolist()))
                    y_all = np.array([y_map.get(pid, np.nan) for pid in pids_all], dtype=np.float32)

                    tr_m = np.isin(pids_all, sp.train_patients) & np.isfinite(y_all)
                    va_m = np.isin(pids_all, sp.val_patients) & np.isfinite(y_all)
                    te_m = np.isin(pids_all, sp.test_patients) & np.isfinite(y_all)

                    if not bool(tr_m.any()) or not bool(va_m.any()) or not bool(te_m.any()):
                        raise SystemExit("No samples available after split for static-only DL training.")

                    X_static = torch.from_numpy(static_train.to_numpy(dtype=np.float32, copy=True))
                    # Dummy lengths (not used in MLP tabular mode)
                    L_static = torch.ones((X_static.shape[0],), dtype=torch.long)

                    X_tr, L_tr, y_tr = X_static[tr_m], L_static[tr_m], torch.from_numpy(y_all[tr_m])
                    X_va, L_va, y_va = X_static[va_m], L_static[va_m], torch.from_numpy(y_all[va_m])
                    X_te, L_te, y_te = X_static[te_m], L_static[te_m], torch.from_numpy(y_all[te_m])

                    input_dim = int(X_static.shape[1])
                    cfg_use = replace(cfg, _dynamic_dim=input_dim)
                    built = build_model(cfg_use)
                    model = built.model

                    from oneehr.modeling.trainer import fit_sequence_model

                    fit = fit_sequence_model(
                        model=model,
                        X_train=X_tr,
                        len_train=L_tr,
                        y_train=y_tr,
                        static_train=None,
                        X_val=X_va,
                        len_val=L_va,
                        y_val=y_va,
                        static_val=None,
                        task=cfg.task,
                        trainer=cfg.trainer,
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
                        out_dir=model_out,
                        model=model,
                        cfg=cfg,
                        feature_columns=list(static_feature_columns),
                        code_vocab=None,
                    )
                else:
                    # Unified DL model construction via build_model().
                    feat_cols = list(dynamic_feature_columns)
                    input_dim = len(feat_cols)
                    if cfg.task.kind == "binary" and len(np.unique(y_train.astype(float))) < 2:
                        rows.append(
                            {
                                "model": model_name,
                                "split": sp.name,
                                "skipped": 1,
                                "reason": "single_class_train",
                            }
                        )
                        continue
                    if cfg.task.kind == "binary" and len(np.unique(y_val.astype(float))) < 2:
                        rows.append(
                            {
                                "model": model_name,
                                "split": sp.name,
                                "skipped": 1,
                                "reason": "single_class_val",
                            }
                        )
                        continue

                    cfg_use = replace(cfg, _dynamic_dim=input_dim)
                    if static_train is not None:
                        cfg_use = replace(cfg_use, _static_dim=int(static_train.shape[1]))
                    built = build_model(cfg_use)
                    model = built.model

                    model_supports_static_branch = hasattr(model, "static_dim") and int(getattr(model, "static_dim", 0)) > 0

                    if cfg.task.prediction_mode == "patient":
                        (
                            y_score,
                            y_test,
                            test_patient_ids,
                            test_logits,
                            val_score,
                            val_logits,
                            y_val_true,
                        ) = _train_sequence_patient_level(
                            model=model,
                            binned=binned,
                            y=y,
                            static=static_train,
                            split=sp,
                            cfg=cfg.trainer,
                            task=cfg.task,
                            model_supports_static_branch=model_supports_static_branch,
                            y_map=dict(zip(X.index.astype(str).tolist(), y.to_numpy().tolist())),
                        )
                    else:
                        (
                            y_score,
                            y_test,
                            test_key_rows,
                            test_logits,
                            val_score,
                            val_logits,
                            y_val_true,
                        ) = _train_sequence_time_level(
                            model=model,
                            binned=binned,
                            labels_df=labels_df,
                            static=static_train,
                            split=sp,
                            cfg=cfg.trainer,
                            task=cfg.task,
                            model_supports_static_branch=model_supports_static_branch,
                        )
                        test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                        test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])

                    model_out = ensure_dir(out_root / "models" / model_name / sp.name)
                    code_vocab = None
                    write_dl_artifacts(
                        out_dir=model_out,
                        model=model,
                        cfg=cfg,
                        feature_columns=feat_cols,
                        code_vocab=code_vocab,
                    )

            # Optional calibration (fit on val split) + threshold selection.
            cal_extra = {}
            if cfg.task.kind == "binary":
                if cfg.model.name in TABULAR_MODELS:
                    val_score = predict_tabular(art, X_val, cfg.task)
                    val_logits = predict_tabular_logits(art, X_val, cfg.task)
                    test_logits = predict_tabular_logits(art, X_test, cfg.task)
                    y_val_true = y_val.astype(float)

                y_score, cal_extra = _maybe_calibrate_and_threshold(
                    cfg0=cfg0,
                    y_val_true=y_val_true.astype(float),
                    y_val_score=val_score.astype(float),
                    y_val_logits=val_logits,
                    y_test_score=y_score.astype(float),
                    y_test_logits=test_logits,
                )

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
            cfg0=cfg0,
            binned=binned,
            labels_df=labels_df,
            patient_index=patient_index,
            out_root=out_root,
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
                X_train=X_train,
                y_train=y_train,
                X_val=None,
                y_val=None,
                task=cfg_fit.task,
                model_cfg=cfg_fit.model.xgboost,
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
                y_true=y_test.astype(float),
                y_pred=y_pred.astype(float),
                task=cfg_fit.task,
                metric=metric_name,
                n=cfg_fit.trainer.bootstrap_n,
                seed=cfg_fit.trainer.seed,
            )
            write_json(
                final_dir / f"test_bootstrap_{model_name}.json",
                {
                    "metric": bs.metric,
                    "n": bs.n,
                    "mean": bs.mean,
                    "ci_low": bs.ci_low,
                    "ci_high": bs.ci_high,
                },
            )

        if cfg_fit.output.save_preds and len(X_test) > 0:
            pd.DataFrame(
                {"patient_id": X_test.index.astype(str), "y_true": y_test, "y_pred": y_pred}
            ).to_parquet(final_dir / f"test_preds_{model_name}.parquet", index=False)

        rows.append({"model": model_name, **metrics})

    if rows:
        write_json(final_dir / "test_summary.json", {"records": rows})
