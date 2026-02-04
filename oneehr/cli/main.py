import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from oneehr.utils.imports import optional_import

from oneehr.config.load import load_experiment_config
from oneehr.data.patient_index import make_patient_index
from oneehr.data.io import load_event_table
from oneehr.data.static_features import build_static_features
from oneehr.data.tabular import make_patient_tabular, make_time_tabular
from oneehr.models.tabular import (
    load_tabular_model,
    predict_tabular,
    predict_tabular_logits,
    save_tabular_model,
    train_tabular_model,
)
from oneehr.data.binning import bin_events
from oneehr.data.labels import normalize_patient_labels, normalize_time_labels, run_label_fn
from oneehr.eval.metrics import binary_metrics, regression_metrics
from oneehr.eval.calibration import (
    binary_brier,
    binary_log_loss,
    calibrate_from_logits,
    calibrate_from_probs,
    select_threshold_f1,
)
from oneehr.eval.bootstrap import bootstrap_metric
from oneehr.data.sequence import build_patient_sequences
from oneehr.data.splits import make_splits
from oneehr.data.postprocess import maybe_fit_transform_postprocess
from oneehr.utils.io import ensure_dir, write_json
from oneehr.eval.tables import summarize_metrics, to_paper_wide_table
from oneehr.hpo.grid import apply_overrides, iter_grid
from oneehr.hpo.runner import select_best_with_trials
from oneehr.modeling.trainer import fit_sequence_model, fit_sequence_model_time
from oneehr.models.registry import build_model
from oneehr.modeling.persistence import write_dl_artifacts
from oneehr.artifacts.materialize import materialize_preprocess_artifacts
from oneehr.artifacts.read import read_run_manifest
from oneehr.artifacts.run_io import RunIO


def _train_sequence_patient_level(
    model,
    binned,
    y,
    static,
    split,
    cfg,
    task,
):
    torch = optional_import("torch")
    if torch is None:
        raise SystemExit("Missing optional dependency: torch. Install it first (e.g. `uv add torch`).")

    from oneehr.data.sequence import pad_sequences

    feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    pids, seqs, lens = build_patient_sequences(binned, feat_cols)
    X_seq = pad_sequences(seqs, lens)
    lens_t = torch.from_numpy(lens)

    y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
    y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)
    pids_arr = np.array(pids, dtype=str)

    train_m = np.isin(pids_arr, split.train_patients)
    val_m = np.isin(pids_arr, split.val_patients)
    test_m = np.isin(pids_arr, split.test_patients)

    X_tr, L_tr, y_tr = X_seq[train_m], lens_t[train_m], torch.from_numpy(y_arr[train_m])
    X_va, L_va, y_va = X_seq[val_m], lens_t[val_m], torch.from_numpy(y_arr[val_m])
    X_te, L_te, y_te = X_seq[test_m], lens_t[test_m], torch.from_numpy(y_arr[test_m])

    S = None
    if static is not None:
        from oneehr.data.sequence import align_static_features

        S_all = align_static_features(
            pids,
            static,
            expected_feature_columns=list(static.columns),
        )
        if S_all is not None:
            S = torch.from_numpy(S_all)
            S_tr, S_va, S_te = S[train_m], S[val_m], S[test_m]
        else:
            S_tr = S_va = S_te = None
    else:
        S_tr = S_va = S_te = None

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

    # Calibration set inference (val)
    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_va is None:
            val_logits = model(X_va, L_va).squeeze(-1).detach().cpu().numpy()
        else:
            val_logits = model(X_va, L_va, S_va).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        val_score = 1.0 / (1.0 + np.exp(-val_logits))
    else:
        val_score = val_logits

    # final evaluation on test
    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_te is None:
            logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
        else:
            logits = model(X_te, L_te, S_te).squeeze(-1).detach().cpu().numpy()
    y_true = y_te.detach().cpu().numpy()
    if task.kind == "binary":
        y_score = 1.0 / (1.0 + np.exp(-logits))
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
):
    torch = optional_import("torch")
    if torch is None:
        raise SystemExit("Missing optional dependency: torch. Install it first (e.g. `uv add torch`).")

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

    if static is not None:
        from oneehr.data.sequence import align_static_features

        S_all = align_static_features(pids, static, expected_feature_columns=list(static.columns))
        if S_all is not None:
            S = torch.from_numpy(S_all)
            S_tr, S_va, S_te = S[train_m], S[val_m], S[test_m]
        else:
            S_tr = S_va = S_te = None
    else:
        S_tr = S_va = S_te = None

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

    # Calibration set inference (val)
    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_va is None:
            val_logits = model(X_va, L_va).squeeze(-1).detach().cpu().numpy()
        else:
            val_logits = model(X_va, L_va, S_va).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        val_score = 1.0 / (1.0 + np.exp(-val_logits))
    else:
        val_score = val_logits
    y_val_np = Y_va.detach().cpu().numpy()
    m_val_np = M_va.detach().cpu().numpy() if M_va is not None else None
    if m_val_np is not None:
        flat = m_val_np.reshape(-1).astype(bool)
        val_score = val_score.reshape(-1)[flat]
        val_logits = val_logits.reshape(-1)[flat]
        y_val_np = y_val_np.reshape(-1)[flat]

    # final evaluation on test
    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_te is None:
            logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
        else:
            logits = model(X_te, L_te, S_te).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        y_score = 1.0 / (1.0 + np.exp(-logits))
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
    """Apply optional calibration using val split, and compute thresholds.

    Returns: (y_test_score_out, extra_metrics)
    """

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

    # Persist calibrator params into metrics (merged output).
    for k, v in params.items():
        extra[f"calibration_{k}"] = float(v)

    if not cfg0.calibration.use_calibrated:
        return y_test_score, extra

    # Apply to test split using same calibrator parameters.
    if method == "temperature":
        t = float(params["temperature"])
        if y_test_logits is not None:
            y_test_cal = 1.0 / (1.0 + np.exp(-(y_test_logits / t)))
        else:
            eps = np.finfo(float).eps
            p = np.clip(y_test_score.astype(float), eps, 1.0 - eps)
            logits = np.log(p / (1.0 - p))
            y_test_cal = 1.0 / (1.0 + np.exp(-(logits / t)))
        return y_test_cal.astype(float), extra

    if method == "platt":
        a = float(params["a"])
        b = float(params["b"])
        if y_test_logits is not None:
            z = a * y_test_logits + b
        else:
            eps = np.finfo(float).eps
            p = np.clip(y_test_score.astype(float), eps, 1.0 - eps)
            logits = np.log(p / (1.0 - p))
            z = a * logits + b
        y_test_cal = 1.0 / (1.0 + np.exp(-z))
        return y_test_cal.astype(float), extra

    raise SystemExit(f"Unsupported calibration.method={method!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oneehr")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Preprocess raw event table into model-ready artifacts")
    p_pre.add_argument("--config", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True)
    p_train.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run directory if it exists",
    )

    p_hpo = sub.add_parser("hpo", help="Run hyperparameter selection on validation")
    p_hpo.add_argument("--config", required=True)

    p_test = sub.add_parser("test", help="Evaluate a trained model on a test dataset")
    p_test.add_argument("--config", required=True)
    p_test.add_argument("--run-dir", required=False, help="Path to a training run directory")
    p_test.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing test output directory if it exists",
    )
    p_test.add_argument(
        "--test-dataset",
        required=False,
        help="Override test dataset path (CSV/XLSX); requires datasets.test in config otherwise",
    )
    p_test.add_argument(
        "--out-dir",
        required=False,
        help=(
            "Directory to write test outputs (metrics/preds). "
            "Defaults to <run_dir>/test_runs/<dataset_stem>"
        ),
    )

    return parser


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


def _run_preprocess(cfg_path: str) -> None:
    cfg = load_experiment_config(cfg_path)
    events = load_event_table(cfg.dataset)
    out_root = cfg.output.root / cfg.output.run_name
    materialize_preprocess_artifacts(events=events, cfg=cfg, out_root=out_root)

    labels_res = run_label_fn(events, cfg)

    if labels_res is not None:
        if cfg.task.prediction_mode == "patient":
            labels = normalize_patient_labels(labels_res.df)
        elif cfg.task.prediction_mode == "time":
            labels = normalize_time_labels(labels_res.df, cfg)
        else:
            raise SystemExit(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")
        ensure_dir(out_root)
        (out_root / "labels.parquet").write_bytes(labels.to_parquet(index=False))


def _run_train(cfg_path: str, force: bool) -> None:
    cfg0 = load_experiment_config(cfg_path)
    train_dataset = cfg0.datasets.train if cfg0.datasets is not None else cfg0.dataset
    events = load_event_table(train_dataset)
    patient_index = make_patient_index(events, cfg0.dataset.time_col, cfg0.dataset.patient_id_col)
    splits = make_splits(patient_index, cfg0.split)

    out_root = cfg0.output.root / cfg0.output.run_name
    if out_root.exists() and not force:
        raise SystemExit(
            f"Run directory already exists: {out_root}. "
            "Choose a new output.run_name or pass --force to overwrite."
        )
    ensure_dir(out_root)

    # Require preprocess-generated manifest (single source of truth for features/artifacts).
    manifest = read_run_manifest(out_root)
    if manifest is None:
        raise SystemExit("Missing run_manifest.json; run `oneehr preprocess` first.")

    labels_res = run_label_fn(events, cfg0)
    labels_df = None
    if labels_res is not None:
        if cfg0.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(labels_res.df)
        elif cfg0.task.prediction_mode == "time":
            labels_df = normalize_time_labels(labels_res.df, cfg0)

    # Unified training: run all splits by default (or a single fold if configured).

    if any(m.name in {"gru", "rnn", "transformer"} for m in cfg0.models):
        torch = optional_import("torch")
        if torch is None:
            raise SystemExit(
                "Missing optional dependency: torch. Install it first (e.g. `uv add torch`)."
            )

    # Delegate to benchmark logic (supports multi-split + optional HPO) but keep command name `train`.
    # If HPO is disabled, it behaves like fixed-hyperparameter training.
    _run_benchmark(cfg_path, force=force)
    return


def _run_test(
    cfg_path: str,
    run_dir: str | None,
    test_dataset: str | None,
    force: bool,
    out_dir: str | None,
) -> None:
    cfg0 = load_experiment_config(cfg_path)
    if run_dir is None:
        run_root = cfg0.output.root / cfg0.output.run_name
    else:
        run_root = Path(run_dir)
    run = RunIO(run_root=Path(run_root))
    manifest = run.require_manifest()
    dynamic_feature_columns = manifest.dynamic_feature_columns()
    static_all, static_feature_columns = run.load_static_all(manifest)

    # Resolve test dataset from CLI override or config.
    if test_dataset is not None:
        ds = replace(
            cfg0.dataset,
            path=Path(test_dataset),
        )
    else:
        if cfg0.datasets is None or cfg0.datasets.test is None:
            raise SystemExit("Missing test dataset. Provide --test-dataset or set [datasets.test] in config.")
        ds = cfg0.datasets.test

    events = load_event_table(ds)
    binned = run.load_binned(manifest)

    labels_res = run_label_fn(events, cfg0)
    labels_df = None
    if labels_res is not None:
        if cfg0.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(labels_res.df)
        else:
            labels_df = normalize_time_labels(labels_res.df, cfg0)

    import pandas as pd

    if cfg0.task.prediction_mode == "patient":
        X, y = run.load_patient_view(manifest)
        key_df = None
        global_mask = None
    elif cfg0.task.prediction_mode == "time":
        X, y, key_df = run.load_time_view(manifest)
        global_mask = None
    else:
        raise SystemExit(f"Unsupported task.prediction_mode={cfg0.task.prediction_mode!r}")

    if out_dir is None:
        out_path = run_root / "test_runs" / ds.path.stem
    else:
        out_path = Path(out_dir)

    if out_path.exists() and force:
        import shutil

        shutil.rmtree(out_path)
    out_dir = ensure_dir(out_path)

    rows = []
    for model_cfg in (cfg0.models or [cfg0.model]):
        model_name = model_cfg.name
        model_dir = run_root / "models" / model_name
        if not model_dir.exists():
            raise SystemExit(f"Missing trained model artifacts: {model_dir}")

        is_tabular = model_name in {"xgboost", "catboost", "rf", "dt", "gbdt"}
        is_dl = not is_tabular

        # Evaluate all trained splits (folds) under this model.
        split_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
        if not split_dirs:
            raise SystemExit(
                f"No split subdirectories found under {model_dir}. "
                "Re-train with updated train command to persist per-split models."
            )

        for sp_dir in split_dirs:
            if is_tabular:
                cols_path = sp_dir / "feature_columns.json"
                if not cols_path.exists():
                    continue
                art = load_tabular_model(sp_dir, task=cfg0.task, kind=model_name)
                y_pred = predict_tabular(art, X, cfg0.task)
            else:
                torch = optional_import("torch")
                if torch is None:
                    raise SystemExit(
                        "Missing optional dependency: torch. Install it first (e.g. `uv add torch`)."
                    )

                import json
                import hashlib

                feat_cols = list(dynamic_feature_columns)
                exp_cols = (
                    (((manifest.data.get("features") or {}).get("dynamic") or {}).get("feature_columns"))
                    or None
                )
                if isinstance(exp_cols, list) and exp_cols and exp_cols != feat_cols:
                    raise SystemExit(
                        "DL test dynamic feature_columns mismatch with run_manifest.json. "
                        "Re-run preprocess/train with consistent features or use a different run dir."
                    )

                meta_path = sp_dir / "model_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    exp_cols = (meta.get("input") or {}).get("feature_columns")
                    exp_sha = (meta.get("input") or {}).get("feature_columns_sha256")
                    if isinstance(exp_cols, list) and exp_cols and exp_cols != feat_cols:
                        raise SystemExit(
                            "DL test feature_columns mismatch with saved model_meta.json. "
                            "Re-run preprocess/train with consistent features or use a different run dir."
                        )
                    if isinstance(exp_sha, str) and exp_sha:
                        norm = "\n".join([c.strip() for c in feat_cols]) + "\n"
                        got_sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                        if got_sha != exp_sha:
                            raise SystemExit(
                                "DL test feature_columns hash mismatch with saved model_meta.json. "
                                "Re-run preprocess/train with consistent features or use a different run dir."
                            )
                pids, seqs, lens = build_patient_sequences(binned, feat_cols)
                from oneehr.data.sequence import pad_sequences, build_time_sequences

                # Build model from config (ensures same architecture); then load weights.
                built = build_model(replace(cfg0, model=model_cfg, models=[model_cfg]))
                model = built.model
                ckpt_path = sp_dir / "state_dict.ckpt"
                if not ckpt_path.exists():
                    raise SystemExit(f"Missing DL checkpoint: {ckpt_path}")
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)
                model.eval()

                if cfg0.task.prediction_mode == "patient":
                    X_seq = pad_sequences(seqs, lens)
                    lens_t = torch.from_numpy(lens)
                    S = None
                    if static_all is not None:
                        from oneehr.data.sequence import align_static_features

                        S_np = align_static_features(
                            pids,
                            static_all,
                            expected_feature_columns=list(static_feature_columns or []),
                        )
                        S = None if S_np is None else torch.from_numpy(S_np)
                    with torch.no_grad():
                        if S is None:
                            logits = model(X_seq, lens_t).squeeze(-1).detach().cpu().numpy()
                        else:
                            logits = model(X_seq, lens_t, S).squeeze(-1).detach().cpu().numpy()
                    if cfg0.task.kind == "binary":
                        y_pred = 1.0 / (1.0 + np.exp(-logits))
                    else:
                        y_pred = logits
                    # Align to X index order
                    import pandas as pd

                    pred_s = pd.Series(y_pred, index=pd.Index(np.array(pids, dtype=str), name="patient_id"))
                    y_pred = pred_s.reindex(X.index.astype(str)).to_numpy()
                else:
                    if labels_df is None:
                        raise SystemExit("DL test for prediction_mode='time' requires labels_df")
                    pids2, time_seqs, seqs2, y_seqs, mask_seqs, lens2 = build_time_sequences(
                        binned,
                        labels_df,
                        feat_cols,
                        label_time_col="bin_time",
                    )
                    X_seq = pad_sequences(seqs2, lens2)
                    lens_t = torch.from_numpy(lens2)
                    S = None
                    if static_all is not None:
                        from oneehr.data.sequence import align_static_features

                        S_np = align_static_features(
                            pids2,
                            static_all,
                            expected_feature_columns=list(static_feature_columns or []),
                        )
                        S = None if S_np is None else torch.from_numpy(S_np)
                    with torch.no_grad():
                        if S is None:
                            logits = model(X_seq, lens_t).squeeze(-1).detach().cpu().numpy()
                        else:
                            logits = model(X_seq, lens_t, S).squeeze(-1).detach().cpu().numpy()
                    if cfg0.task.kind == "binary":
                        probs = 1.0 / (1.0 + np.exp(-logits))
                    else:
                        probs = logits
                    # Flatten by mask to match make_time_tabular(label-joined) ordering
                    key_rows = []
                    preds_flat = []
                    for pid, t, m, pr in zip(pids2, time_seqs, mask_seqs, probs, strict=True):
                        for tt, mm, vv in zip(t, m, pr, strict=True):
                            if bool(mm):
                                key_rows.append((str(pid), tt))
                                preds_flat.append(float(vv))
                    pred_df = pd.DataFrame(key_rows, columns=["patient_id", "bin_time"])
                    pred_df["y_pred"] = np.array(preds_flat)
                    if key_df is None:
                        raise SystemExit("Internal error: missing key_df for time mode")
                    merged = key_df.merge(pred_df, on=["patient_id", "bin_time"], how="left")
                    y_pred = merged["y_pred"].to_numpy()

            y_true_np = y.to_numpy().astype(float)
            if global_mask is not None:
                y_true_np = y_true_np.astype(float)
                y_pred = np.asarray(y_pred)

            if cfg0.task.kind == "binary":
                metrics = binary_metrics(y_true_np.astype(float), np.asarray(y_pred).astype(float)).metrics
            else:
                metrics = regression_metrics(y_true_np.astype(float), np.asarray(y_pred).astype(float)).metrics

            tag = f"{model_name}__{sp_dir.name}"
            write_json(out_dir / f"metrics_{tag}.json", metrics)
            if cfg0.output.save_preds:
                pd.DataFrame(
                    {"patient_id": X.index.astype(str), "y_true": y.to_numpy(), "y_pred": y_pred}
                ).to_parquet(out_dir / f"preds_{tag}.parquet", index=False)
            rows.append({"model": model_name, "split": sp_dir.name, **metrics})

    write_json(
        out_dir / "summary.json",
        {
            "task": {"kind": str(cfg0.task.kind), "prediction_mode": str(cfg0.task.prediction_mode)},
            "records": rows,
        },
    )


def _run_benchmark(cfg_path: str, *, force: bool = False) -> None:
    cfg0 = load_experiment_config(cfg_path)
    train_dataset = cfg0.datasets.train if cfg0.datasets is not None else cfg0.dataset
    events = load_event_table(train_dataset)
    static_raw = build_static_features(events, cfg0.dataset, cfg0.static_features)
    patient_index = make_patient_index(events, cfg0.dataset.time_col, cfg0.dataset.patient_id_col)
    splits = make_splits(patient_index, cfg0.split)

    labels_res = run_label_fn(events, cfg0)
    labels_df = None
    if labels_res is not None:
        if cfg0.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(labels_res.df)
        elif cfg0.task.prediction_mode == "time":
            labels_df = normalize_time_labels(labels_res.df, cfg0)

    out_root = cfg0.output.root / cfg0.output.run_name
    manifest = read_run_manifest(out_root)
    if manifest is None:
        raise SystemExit("Missing run_manifest.json; run `oneehr preprocess` first.")
    import pandas as pd
    binned_path = (manifest.data.get("artifacts") or {}).get("binned_parquet_path")
    if not isinstance(binned_path, str) or not binned_path:
        raise SystemExit("Missing binned_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
    binned = pd.read_parquet(out_root / binned_path)

    if cfg0.task.prediction_mode == "patient":
        pt_path = (manifest.data.get("artifacts") or {}).get("patient_tabular_parquet_path")
        if not isinstance(pt_path, str) or not pt_path:
            raise SystemExit("Missing patient_tabular_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        dfp = pd.read_parquet(out_root / pt_path)
        if "patient_id" not in dfp.columns or "label" not in dfp.columns:
            raise SystemExit("Invalid patient_tabular.parquet: missing patient_id/label.")
        dfp = dfp.dropna(subset=["label"]).reset_index(drop=True)
        X = dfp.drop(columns=["label"]).set_index("patient_id")
        y = dfp["label"]
        key = None
        global_mask = None
    elif cfg0.task.prediction_mode == "time":
        tm_path = (manifest.data.get("artifacts") or {}).get("time_tabular_parquet_path")
        if not isinstance(tm_path, str) or not tm_path:
            raise SystemExit("Missing time_tabular_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        dft = pd.read_parquet(out_root / tm_path)
        required = {"patient_id", "bin_time", "label"}
        missing = [c for c in required if c not in dft.columns]
        if missing:
            raise SystemExit(f"Invalid time_tabular.parquet: missing columns {missing}")
        key = dft[["patient_id", "bin_time"]].reset_index(drop=True)
        y = dft["label"].reset_index(drop=True)
        X = dft.drop(columns=["patient_id", "bin_time", "label"]).reset_index(drop=True)
        global_mask = None
    else:
        raise SystemExit(f"Unsupported task.prediction_mode={cfg0.task.prediction_mode!r}")
    dynamic_feature_columns = manifest.dynamic_feature_columns()

    static_all = None
    static_feature_columns = None
    if cfg0.static_features.enabled:
        st_path = manifest.static_matrix_path()
        if st_path is None:
            raise SystemExit(
                "Static features enabled but static matrix not found in run_manifest.json. "
                "Re-run `oneehr preprocess`."
            )
        static_all = pd.read_parquet(out_root / st_path)
        static_feature_columns = manifest.static_feature_columns()
        if list(static_all.columns) != list(static_feature_columns):
            raise SystemExit(
                "Static feature_columns mismatch with run_manifest.json. "
                "Re-run `oneehr preprocess`."
            )

    models = cfg0.models or [cfg0.model]
    if any(m.name in {"gru", "rnn", "transformer"} for m in models):
        torch = optional_import("torch")
        if torch is None:
            raise SystemExit(
                "Missing optional dependency: torch. Install it first (e.g. `uv add torch`)."
            )

    rows = []
    run_records: list[dict[str, object]] = []
    if out_root.exists() and force:
        import shutil

        shutil.rmtree(out_root)
        ensure_dir(out_root)

    # Single-process only.

    for model_cfg in models:
        cfg_model = replace(cfg0, model=model_cfg, models=[model_cfg])
        model_name = cfg_model.model.name
        if model_name in cfg0.hpo_by_model:
            cfg_model = replace(cfg_model, hpo=cfg0.hpo_by_model[model_name])

        _warn_unused_hpo_overrides(model_name, list(iter_grid(cfg_model.hpo)))

        # Optional: run HPO once on a selected split, then reuse best overrides across all splits.
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

        # Optional: cross-split HPO selection.
        # For each hyperparameter override, evaluate on every split's validation set and
        # pick the override with the best mean score across splits.
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

                    # Only implement XGBoost for now (consistent with HPO/test support).
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
            # Ensure no NaNs in labels (can happen if split includes unlabeled patients).
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

            # Post-merge preprocessing pipeline (fit on train split only).
            X_train, X_val, X_test, fitted_post = maybe_fit_transform_postprocess(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                pipeline=cfg0.preprocess.pipeline,
            )

            # Pass full static matrix; batch builders align to patient_id order.
            static_train = static_all

            # Select best hyperparameters using validation.
            def _eval_trial(cfg) -> tuple[float, dict[str, float]] | None:
                # If users configured HPO metric incompatible with current trial type,
                # fall back to the trainer monitor for DL trials.
                hpo_metric = cfg.hpo.metric

                if cfg.model.name in {"xgboost", "catboost", "rf", "dt", "gbdt"}:
                    if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                        return None
                    model_cfg = getattr(cfg.model, cfg.model.name)
                    art = train_tabular_model(
                        model_name=cfg.model.name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        task=cfg.task,
                        model_cfg=model_cfg,
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

                torch = optional_import("torch")
                if torch is None:
                    raise SystemExit("DL HPO requires torch")

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
                    cfg_use = replace(cfg, preprocess=replace(cfg.preprocess, top_k_codes=input_dim))
                    built = build_model(cfg_use)
                    if built.kind != "dl":
                        return None
                    model = built.model

                    fit = fit_sequence_model(
                        model=model,
                        X_train=X_tr,
                        len_train=L_tr,
                        y_train=y_tr,
                        X_val=X_va,
                        len_val=L_va,
                        y_val=y_va,
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
                    Y_seq = pad_sequences([y[:, None] for y in y_seqs], lens).squeeze(-1)
                    M_seq = pad_sequences([m[:, None] for m in mask_seqs], lens).squeeze(-1)
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
                    cfg_use = replace(cfg, preprocess=replace(cfg.preprocess, top_k_codes=input_dim))
                    built = build_model(cfg_use)
                    if built.kind != "dl":
                        return None
                    model = built.model

                    fit = fit_sequence_model_time(
                        model=model,
                        X_train=X_tr,
                        len_train=L_tr,
                        y_train=Y_tr,
                        mask_train=M_tr,
                        X_val=X_va,
                        len_val=L_va,
                        y_val=Y_va,
                        mask_val=M_va,
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

                # Persist trial results per split.
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
            if cfg.model.name in {"xgboost", "catboost", "rf", "dt", "gbdt"}:
                if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                    # Keep a row for completeness, but mark as skipped.
                    row = {
                        "model": model_name,
                        "split": sp.name,
                        "skipped": 1,
                        "reason": "single_class_train",
                    }
                    rows.append(row)
                    continue

                model_cfg = getattr(cfg.model, cfg.model.name)
                art = train_tabular_model(
                    model_name=cfg.model.name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    task=cfg.task,
                    model_cfg=model_cfg,
                )
                model_out = ensure_dir(out_root / "models" / model_name / sp.name)
                save_tabular_model(art, model_out)
                y_score = predict_tabular(art, X_test, cfg.task)
                if fitted_post is not None:
                    pp_dir = ensure_dir(out_root / "preprocess" / sp.name)
                    write_json(pp_dir / "pipeline.json", {"pipeline": fitted_post.pipeline})
                # Static features are persisted at preprocess time via run_manifest.json and
                # features/static/static_all.parquet.
            else:
                feat_cols = list(dynamic_feature_columns)
                input_dim = len(feat_cols)
                if cfg.model.name == "gru":
                    from oneehr.models.gru import GRUModel, GRUTimeModel

                    if cfg.task.prediction_mode == "patient":
                        model = GRUModel(
                            input_dim=input_dim,
                            hidden_dim=cfg.model.gru.hidden_dim,
                            out_dim=1,
                            num_layers=cfg.model.gru.num_layers,
                            dropout=cfg.model.gru.dropout,
                        )
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
                        )
                    else:
                        model = GRUTimeModel(
                            input_dim=input_dim,
                            hidden_dim=cfg.model.gru.hidden_dim,
                            out_dim=1,
                            num_layers=cfg.model.gru.num_layers,
                            dropout=cfg.model.gru.dropout,
                        )
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
                        )
                        test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                        test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])
                elif cfg.model.name == "rnn":
                    from oneehr.models.rnn import RNNModel, RNNTimeModel

                    if cfg.task.prediction_mode == "patient":
                        model = RNNModel(
                            input_dim=input_dim,
                            hidden_dim=cfg.model.rnn.hidden_dim,
                            out_dim=1,
                            num_layers=cfg.model.rnn.num_layers,
                            dropout=cfg.model.rnn.dropout,
                            bidirectional=cfg.model.rnn.bidirectional,
                            nonlinearity=cfg.model.rnn.nonlinearity,
                        )
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
                        )
                    else:
                        model = RNNTimeModel(
                            input_dim=input_dim,
                            hidden_dim=cfg.model.rnn.hidden_dim,
                            out_dim=1,
                            num_layers=cfg.model.rnn.num_layers,
                            dropout=cfg.model.rnn.dropout,
                            bidirectional=cfg.model.rnn.bidirectional,
                            nonlinearity=cfg.model.rnn.nonlinearity,
                        )
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
                        )
                        test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                        test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])
                else:
                    from oneehr.models.transformer import TransformerModel, TransformerTimeModel

                    if cfg.task.prediction_mode == "patient":
                        model = TransformerModel(
                            input_dim=input_dim,
                            d_model=cfg.model.transformer.d_model,
                            out_dim=1,
                            nhead=cfg.model.transformer.nhead,
                            num_layers=cfg.model.transformer.num_layers,
                            dim_feedforward=cfg.model.transformer.dim_feedforward,
                            dropout=cfg.model.transformer.dropout,
                            pooling=cfg.model.transformer.pooling,
                        )
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
                        )
                    else:
                        model = TransformerTimeModel(
                            input_dim=input_dim,
                            d_model=cfg.model.transformer.d_model,
                            out_dim=1,
                            nhead=cfg.model.transformer.nhead,
                            num_layers=cfg.model.transformer.num_layers,
                            dim_feedforward=cfg.model.transformer.dim_feedforward,
                            dropout=cfg.model.transformer.dropout,
                        )
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
                        )
                        test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                        test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])

                # Persist DL model artifacts for reproducible test-time loading.
                model_out = ensure_dir(out_root / "models" / model_name / sp.name)
                code_vocab = None
                write_dl_artifacts(
                    out_dir=model_out,
                    model=model,
                    cfg=cfg,
                    feature_columns=feat_cols,
                    code_vocab=code_vocab,
                )
                # Static features are persisted at preprocess time via run_manifest.json and
                # features/static/static_all.parquet.

            # Optional calibration (fit on val split) + threshold selection.
            cal_extra = {}
            if cfg.task.kind == "binary":
                if cfg.model.name in {"xgboost", "catboost", "rf", "dt", "gbdt"}:
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

            # Guard against any remaining unlabeled rows in test (should be rare).
            finite_mask = np.isfinite(y_test.astype(float))
            y_test_eval = y_test[finite_mask]
            y_score_eval = np.asarray(y_score)[finite_mask]
            if y_test_eval.size == 0:
                # No labeled samples in this split/test partition.
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

    # Persist best overrides per split for reproducibility.
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

    if cfg0.trainer.final_model_source not in {"refit", "best_split"}:
        raise SystemExit("trainer.final_model_source must be 'refit' or 'best_split'")

    boundary = pd.to_datetime(cfg0.split.time_boundary, errors="raise")
    pid = patient_index[["patient_id", "max_time"]].copy()
    pid["patient_id"] = pid["patient_id"].astype(str)
    pre_patients = pid[pid["max_time"] < boundary]["patient_id"].to_numpy().astype(str)
    post_patients = pid[pid["max_time"] >= boundary]["patient_id"].to_numpy().astype(str)

    if cfg0.task.prediction_mode != "patient":
        raise SystemExit("final prospective eval currently supports prediction_mode='patient' only")

    X, y0 = make_patient_tabular(binned)
    if labels_df is not None:
        y = labels_df.set_index("patient_id")["label"].reindex(X.index.astype(str))
    else:
        y = y0

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
        # Never select based on test performance (leakage). Use validation signal instead.
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
            # Fall back to task defaults if the configured metric isn't present.
            metric = "auroc" if cfg0.task.kind == "binary" else "rmse"
        for model_name, dfm in summary.groupby("model"):
            dfm2 = dfm[dfm.get("skipped", 0) == 0].copy()
            if dfm2.empty or metric not in dfm2.columns:
                continue
            if cfg0.task.kind == "binary":
                best_row = dfm2.sort_values(metric, ascending=False).iloc[0]
            else:
                best_row = dfm2.sort_values(metric, ascending=True).iloc[0]
            best_split_by_model[str(model_name)] = str(best_row["split"])

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


def _run_hpo(cfg_path: str) -> None:
    """Run a pure HPO selection on the first split (validation-driven).

    Intended for quick iteration; `benchmark` already runs per-split selection.
    """

    cfg0 = load_experiment_config(cfg_path)
    events = load_event_table(cfg0.dataset)
    patient_index = make_patient_index(events, cfg0.dataset.time_col, cfg0.dataset.patient_id_col)
    splits = make_splits(patient_index, cfg0.split)
    sp = splits[0]

    out_root = cfg0.output.root / cfg0.output.run_name
    manifest = read_run_manifest(out_root)
    if manifest is None:
        raise SystemExit("Missing run_manifest.json; run `oneehr preprocess` first.")
    import pandas as pd

    binned_path = (manifest.data.get("artifacts") or {}).get("binned_parquet_path")
    if not isinstance(binned_path, str) or not binned_path:
        raise SystemExit("Missing binned_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
    binned = pd.read_parquet(out_root / binned_path)

    labels_res = run_label_fn(events, cfg0)
    labels_df = None
    if labels_res is not None:
        if cfg0.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(labels_res.df)
        else:
            labels_df = normalize_time_labels(labels_res.df, cfg0)

    if cfg0.task.prediction_mode != "patient":
        raise SystemExit("hpo currently supports prediction_mode='patient' only")

    pt_path = (manifest.data.get("artifacts") or {}).get("patient_tabular_parquet_path")
    if not isinstance(pt_path, str) or not pt_path:
        raise SystemExit("Missing patient_tabular_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
    dfp = pd.read_parquet(out_root / pt_path).dropna(subset=["label"]).reset_index(drop=True)
    X = dfp.drop(columns=["label"]).set_index("patient_id")
    y = dfp["label"]
    train_mask = X.index.astype(str).isin(sp.train_patients)
    val_mask = X.index.astype(str).isin(sp.val_patients)
    X_train, y_train = X.loc[train_mask], y.loc[train_mask].to_numpy()
    X_val, y_val = X.loc[val_mask], y.loc[val_mask].to_numpy()

    def _better(mode: str, a: float, b: float) -> bool:
        return a < b if mode == "min" else a > b

    models = cfg0.models or [cfg0.model]
    out_root = cfg0.output.root / cfg0.output.run_name
    ensure_dir(out_root)

    for model_cfg in models:
        cfg_model = replace(cfg0, model=model_cfg, models=[model_cfg])
        model_name = cfg_model.model.name
        if model_name in cfg0.hpo_by_model:
            cfg_model = replace(cfg_model, hpo=cfg0.hpo_by_model[model_name])

        if cfg_model.model.name != "xgboost":
            raise SystemExit("hpo currently supports model.name='xgboost' only.")

        best_overrides = None
        best_val = None

        for overrides in iter_grid(cfg_model.hpo):
            cfg = apply_overrides(cfg_model, overrides)
            if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                continue
            art = train_tabular_model(
                model_name="xgboost",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task=cfg.task,
                model_cfg=cfg.model.xgboost,
            )
            y_val_score = predict_tabular(art, X_val, cfg.task)
            if cfg.task.kind == "binary":
                vm = binary_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
                val_score = (
                    float(vm["auroc"]) if cfg.hpo.metric in {"val_auroc", "auroc"} else float(vm["auprc"])
                )
            else:
                vm = regression_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
                val_score = (
                    float(vm["rmse"]) if cfg.hpo.metric in {"val_rmse", "rmse"} else float(vm["mae"])
                )

            if best_val is None or _better(cfg.hpo.mode, val_score, best_val):
                best_val = val_score
                best_overrides = dict(overrides)

        write_json(
            out_root / f"hpo_best_{model_name}.json",
            {"best_overrides": best_overrides, "best_val": best_val},
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        _run_preprocess(args.config)
        return
    if args.command == "train":
        _run_train(args.config, force=bool(args.force))
        return
    if args.command == "hpo":
        _run_hpo(args.config)
        return
    if args.command == "test":
        _run_test(
            args.config,
            args.run_dir,
            args.test_dataset,
            force=bool(args.force),
            out_dir=args.out_dir,
        )
        return
    raise SystemExit(f"Unknown command: {args.command}")
