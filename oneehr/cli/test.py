"""oneehr test subcommand."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from oneehr.eval.calibration import sigmoid
import torch
from oneehr.utils.io import ensure_dir, write_json


def run_test(
    cfg_path: str,
    run_dir: str | None,
    test_dataset: str | None,
    force: bool,
    out_dir: str | None,
) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.cli._common import resolve_run_root

    cfg0 = load_experiment_config(cfg_path)
    run_root = resolve_run_root(cfg0, run_dir)
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}")

    # Mode detection: external test dataset vs self-split evaluation.
    is_external = test_dataset is not None
    if not is_external and cfg0.datasets is not None and cfg0.datasets.test is not None:
        is_external = True

    if is_external:
        _run_external_test(
            cfg0=cfg0,
            cfg_path=cfg_path,
            run_root=run_root,
            test_dataset=test_dataset,
            force=force,
            out_dir=out_dir,
        )
    else:
        _run_self_split_test(
            cfg0=cfg0,
            run_root=run_root,
            force=force,
            out_dir=out_dir,
        )


# ─── External test dataset mode ──────────────────────────────────────────────


def _run_external_test(
    *,
    cfg0,
    cfg_path: str,
    run_root: Path,
    test_dataset: str | None,
    force: bool,
    out_dir: str | None,
) -> None:
    import pandas as pd

    from oneehr.config.load import load_experiment_config
    from oneehr.data.io import load_dynamic_table, load_label_table, load_static_table
    from oneehr.artifacts.inference import materialize_test_views
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.models.tabular import load_tabular_model, predict_tabular
    from oneehr.models.registry import build_model
    from oneehr.cli._common import require_manifest

    manifest = require_manifest(run_root)

    if test_dataset is not None:
        test_cfg = load_experiment_config(test_dataset)
        test_ds = test_cfg.datasets.test if test_cfg.datasets is not None else test_cfg.dataset
    elif cfg0.datasets is not None and cfg0.datasets.test is not None:
        test_ds = cfg0.datasets.test
    else:
        test_ds = cfg0.dataset

    dynamic = load_dynamic_table(test_ds.dynamic)
    static = load_static_table(test_ds.static)
    label = load_label_table(test_ds.label)

    labels_fn = cfg0.labels.fn if cfg0.labels.fn else None
    views = materialize_test_views(
        manifest=manifest,
        dynamic=dynamic,
        static=static,
        label=label,
        labels_fn=labels_fn,
    )
    X = views.X
    y = views.y

    feat_cols = manifest.dynamic_feature_columns()
    mode = str((manifest.data.get("task") or {}).get("prediction_mode", "patient"))
    task_kind = str((manifest.data.get("task") or {}).get("kind", "binary"))

    output = Path(out_dir) if out_dir else run_root / "test_runs"
    if output.exists() and not force:
        raise SystemExit(f"Output directory already exists: {output}. Use --force to overwrite.")
    ensure_dir(output)

    from oneehr.models.constants import TABULAR_MODELS
    models_dir = run_root / "models"
    if not models_dir.exists():
        print(f"No models directory found at {models_dir}. Nothing to test.", file=sys.stderr)
        ensure_dir(output)
        write_json(output / "test_summary.json", {"records": []})
        return

    rows = []
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name

            if model_name in TABULAR_MODELS:
                model_path = split_dir / "model.json"
                if not model_path.exists():
                    model_path = split_dir / "model.cbm"
                if not model_path.exists():
                    model_path = split_dir / "model.pkl"
                if not model_path.exists():
                    model_path = split_dir / "model.joblib"
                if not model_path.exists():
                    continue
                from oneehr.config.schema import TaskConfig

                task_cfg = TaskConfig(kind=task_kind, prediction_mode=mode)
                art = load_tabular_model(split_dir, task=task_cfg, kind=model_name)
                y_pred = predict_tabular(art, X, task_cfg)
            else:
                ckpt = split_dir / "state_dict.ckpt"
                meta_path = split_dir / "model_meta.json"
                if not ckpt.exists() or not meta_path.exists():
                    continue

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stored_feat_cols = (meta.get("input") or {}).get("feature_columns", feat_cols)
                input_dim = int(len(stored_feat_cols))

                from oneehr.data.sequence import build_patient_sequences, build_time_sequences, pad_sequences

                model_cfg_obj = cfg0.model
                for mc in (cfg0.models or [cfg0.model]):
                    if mc.name == model_name:
                        model_cfg_obj = mc
                        break

                cfg_use = replace(
                    cfg0,
                    model=model_cfg_obj,
                    _dynamic_dim=input_dim,
                )
                st_cols = manifest.static_feature_columns()
                if st_cols:
                    cfg_use = replace(cfg_use, _static_dim=len(st_cols))
                built = build_model(cfg_use)
                model = built.model
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                model.eval()
                from oneehr.data.features import has_static_branch
                model_supports_static_branch = has_static_branch(model)

                if mode == "patient":
                    pids, seqs, lens = build_patient_sequences(views.binned, stored_feat_cols)
                    X_seq = pad_sequences(seqs, lens)
                    lens_t = torch.from_numpy(lens)

                    with torch.no_grad():
                        if model_supports_static_branch and st_cols:
                            static_mat = (
                                views.X.reindex(index=np.array(pids, dtype=str))
                                .loc[:, st_cols]
                                .to_numpy(dtype=np.float32, copy=True)
                            )
                            logits = model(X_seq, lens_t, torch.from_numpy(static_mat)).squeeze(-1).detach().cpu().numpy()
                        else:
                            logits = model(X_seq, lens_t).squeeze(-1).detach().cpu().numpy()
                    if task_kind == "binary":
                        y_pred_all = sigmoid(logits)
                    else:
                        y_pred_all = logits

                    pid_arr = np.array(pids, dtype=str)
                    y_pred_map = dict(zip(pid_arr.tolist(), y_pred_all.tolist()))
                    X_pids = X.index.astype(str) if hasattr(X.index, "astype") else X.index
                    y_pred = np.array([y_pred_map.get(str(pid), float("nan")) for pid in X_pids])
                else:
                    if views.labels_df is None:
                        continue
                    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
                        views.binned,
                        views.labels_df,
                        stored_feat_cols,
                        label_time_col="bin_time",
                    )
                    X_seq = pad_sequences(seqs, lens)
                    Y_seq = pad_sequences([yy[:, None] for yy in y_seqs], lens).squeeze(-1)
                    M_seq = pad_sequences([mm[:, None] for mm in mask_seqs], lens).squeeze(-1)
                    lens_t = torch.from_numpy(lens)

                    with torch.no_grad():
                        if model_supports_static_branch and st_cols:
                            # Align static by patient_id order (pids); models accept patient-level static.
                            static_mat = (
                                views.X.reindex(index=np.array(pids, dtype=str))
                                .loc[:, st_cols]
                                .to_numpy(dtype=np.float32, copy=True)
                            )
                            logits = model(X_seq, lens_t, torch.from_numpy(static_mat)).squeeze(-1).detach().cpu().numpy()
                        else:
                            logits = model(X_seq, lens_t).squeeze(-1).detach().cpu().numpy()
                    if task_kind == "binary":
                        y_pred_all = sigmoid(logits)
                    else:
                        y_pred_all = logits

                    m_flat = M_seq.detach().cpu().numpy().reshape(-1).astype(bool) if hasattr(M_seq, "detach") else M_seq.reshape(-1).astype(bool)
                    y_pred = y_pred_all.reshape(-1)[m_flat]

            if y is None:
                write_json(
                    output / f"preds_{model_name}_{split_name}.json",
                    {"model": model_name, "split": split_name, "y_pred": y_pred.tolist()},
                )
                continue

            y_true = np.asarray(y, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[finite]
            y_pred = y_pred[finite]
            if y_true.size == 0:
                continue

            if task_kind == "binary":
                metrics = binary_metrics(y_true, y_pred).metrics
            else:
                metrics = regression_metrics(y_true, y_pred).metrics

            write_json(output / f"metrics_{model_name}_{split_name}.json", metrics)

            preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            preds_df.to_parquet(output / f"preds_{model_name}_{split_name}.parquet", index=False)

            rows.append({"model": model_name, "split": split_name, **metrics})

    # Always write a summary file so downstream tooling/tests have a stable
    # artifact to read, even if some models produced no evaluable rows.
    write_json(output / "test_summary.json", {"records": rows})
    if rows:
        pd.DataFrame(rows).to_csv(output / "test_summary.csv", index=False)


# ─── Self-split mode (no external test dataset) ──────────────────────────────


def _run_self_split_test(
    *,
    cfg0,
    run_root: Path,
    force: bool,
    out_dir: str | None,
) -> None:
    """Evaluate trained models on held-out test patients from saved splits.

    This avoids data leakage by filtering preprocessed artifacts to test
    patients only, rather than re-evaluating on the full training dataset.
    """
    import pandas as pd

    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.models.tabular import load_tabular_model, predict_tabular
    from oneehr.models.registry import build_model
    from oneehr.data.postprocess import transform_postprocess_pipeline
    from oneehr.data.splits import load_splits
    from oneehr.artifacts.run_io import RunIO
    from oneehr.cli._common import require_manifest

    manifest = require_manifest(run_root)
    run = RunIO(run_root=run_root)

    feat_cols = manifest.dynamic_feature_columns()
    mode = str((manifest.data.get("task") or {}).get("prediction_mode", "patient"))
    task_kind = str((manifest.data.get("task") or {}).get("kind", "binary"))

    output = Path(out_dir) if out_dir else run_root / "test_runs"
    if output.exists() and not force:
        raise SystemExit(f"Output directory already exists: {output}. Use --force to overwrite.")
    ensure_dir(output)

    # Load saved splits.
    split_dir = run_root / "splits"
    saved_splits = load_splits(split_dir)
    if not saved_splits:
        # Fallback: recompute splits from config with warning.
        print(
            f"WARNING: No saved splits found at {split_dir}. "
            "Recomputing from config. Upgrade by re-running `oneehr train`.",
            file=sys.stderr,
        )
        from oneehr.data.io import load_dynamic_table_optional, load_static_table
        from oneehr.data.patient_index import make_patient_index, make_patient_index_from_static
        from oneehr.data.splits import make_splits

        train_dataset = cfg0.datasets.train if cfg0.datasets is not None else cfg0.dataset
        dynamic_raw = load_dynamic_table_optional(train_dataset.dynamic)
        static_raw = load_static_table(train_dataset.static)
        if dynamic_raw is not None:
            patient_index = make_patient_index(dynamic_raw, "event_time", "patient_id")
        elif static_raw is not None:
            patient_index = make_patient_index_from_static(static_raw, patient_id_col="patient_id")
        else:
            raise SystemExit("Cannot recompute splits: no dynamic or static dataset available.")
        saved_splits = make_splits(patient_index, cfg0.split)

    split_lookup = {sp.name: sp for sp in saved_splits}

    # Load preprocessed artifacts.
    if mode == "patient":
        X_all, y_all = run.load_patient_view(manifest)
    elif mode == "time":
        X_all, y_all, key_all = run.load_time_view(manifest)
    else:
        raise SystemExit(f"Unsupported task.prediction_mode={mode!r}")

    binned_all = run.load_binned(manifest)
    labels_df = run.load_labels(manifest)

    static_all, static_feature_columns = run.load_static_all(manifest)

    from oneehr.models.constants import TABULAR_MODELS
    models_dir = run_root / "models"
    if not models_dir.exists():
        print(f"No models directory found at {models_dir}. Nothing to test.", file=sys.stderr)
        write_json(output / "test_summary.json", {"records": []})
        return

    rows = []
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for split_dir_iter in sorted(model_dir.iterdir()):
            if not split_dir_iter.is_dir():
                continue
            split_name = split_dir_iter.name

            sp = split_lookup.get(split_name)
            if sp is None:
                print(
                    f"WARNING: No split definition found for '{split_name}', skipping.",
                    file=sys.stderr,
                )
                continue

            test_pids = set(sp.test_patients.astype(str).tolist())
            if not test_pids:
                continue

            # Filter to test patients only.
            if mode == "patient":
                test_mask = X_all.index.astype(str).isin(test_pids)
                X_test = X_all.loc[test_mask].copy()
                y_test = y_all.loc[test_mask].to_numpy()
            else:
                assert key_all is not None
                test_mask = key_all["patient_id"].astype(str).isin(test_pids)
                X_test = X_all.loc[test_mask].copy()
                y_test = y_all.loc[test_mask].to_numpy()
                key_test = key_all.loc[test_mask].reset_index(drop=True)

            if len(X_test) == 0:
                continue

            # Apply fitted postprocess pipeline (if saved during training).
            fitted_post = run.load_fitted_postprocess(split_name)
            if fitted_post is not None:
                X_test = transform_postprocess_pipeline(X_test, fitted_post)

            # Join static features for tabular models.
            if model_name in TABULAR_MODELS and static_all is not None and static_feature_columns is not None:
                overlap = [c for c in static_feature_columns if c in X_test.columns]
                static_use = static_all.drop(columns=overlap, errors="ignore")
                X_test = X_test.join(static_use, how="left").fillna(0.0)

            from oneehr.config.schema import TaskConfig
            task_cfg = TaskConfig(kind=task_kind, prediction_mode=mode)

            if model_name in TABULAR_MODELS:
                model_path = split_dir_iter / "model.json"
                if not model_path.exists():
                    model_path = split_dir_iter / "model.cbm"
                if not model_path.exists():
                    model_path = split_dir_iter / "model.pkl"
                if not model_path.exists():
                    model_path = split_dir_iter / "model.joblib"
                if not model_path.exists():
                    continue

                art = load_tabular_model(split_dir_iter, task=task_cfg, kind=model_name)
                y_pred = predict_tabular(art, X_test, task_cfg)
            else:
                ckpt = split_dir_iter / "state_dict.ckpt"
                meta_path = split_dir_iter / "model_meta.json"
                if not ckpt.exists() or not meta_path.exists():
                    continue

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stored_feat_cols = (meta.get("input") or {}).get("feature_columns", feat_cols)
                input_dim = int(len(stored_feat_cols))

                from oneehr.data.sequence import build_patient_sequences, build_time_sequences, pad_sequences

                model_cfg_obj = cfg0.model
                for mc in (cfg0.models or [cfg0.model]):
                    if mc.name == model_name:
                        model_cfg_obj = mc
                        break

                cfg_use = replace(
                    cfg0,
                    model=model_cfg_obj,
                    _dynamic_dim=input_dim,
                )
                st_cols = manifest.static_feature_columns()
                if st_cols:
                    cfg_use = replace(cfg_use, _static_dim=len(st_cols))
                built = build_model(cfg_use)
                model_nn = built.model
                model_nn.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                model_nn.eval()
                from oneehr.data.features import has_static_branch
                model_supports_static_branch = has_static_branch(model_nn)

                # Filter binned to test patients for sequence building.
                binned_test = binned_all[binned_all["patient_id"].astype(str).isin(test_pids)].copy()

                if mode == "patient":
                    pids, seqs, lens = build_patient_sequences(binned_test, stored_feat_cols)
                    X_seq = pad_sequences(seqs, lens)
                    lens_t = torch.from_numpy(lens)

                    with torch.no_grad():
                        if model_supports_static_branch and st_cols:
                            static_mat = (
                                X_all.reindex(index=np.array(pids, dtype=str))
                                .loc[:, st_cols]
                                .to_numpy(dtype=np.float32, copy=True)
                            )
                            logits = model_nn(X_seq, lens_t, torch.from_numpy(static_mat)).squeeze(-1).detach().cpu().numpy()
                        else:
                            logits = model_nn(X_seq, lens_t).squeeze(-1).detach().cpu().numpy()
                    if task_kind == "binary":
                        y_pred_all = sigmoid(logits)
                    else:
                        y_pred_all = logits

                    pid_arr = np.array(pids, dtype=str)
                    y_pred_map = dict(zip(pid_arr.tolist(), y_pred_all.tolist()))
                    X_pids = X_test.index.astype(str) if hasattr(X_test.index, "astype") else X_test.index
                    y_pred = np.array([y_pred_map.get(str(pid), float("nan")) for pid in X_pids])
                else:
                    if labels_df is None:
                        continue
                    # Filter labels_df to test patients.
                    labels_test = labels_df[labels_df["patient_id"].astype(str).isin(test_pids)].copy()
                    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
                        binned_test,
                        labels_test,
                        stored_feat_cols,
                        label_time_col="bin_time",
                    )
                    X_seq = pad_sequences(seqs, lens)
                    Y_seq = pad_sequences([yy[:, None] for yy in y_seqs], lens).squeeze(-1)
                    M_seq = pad_sequences([mm[:, None] for mm in mask_seqs], lens).squeeze(-1)
                    lens_t = torch.from_numpy(lens)

                    with torch.no_grad():
                        if model_supports_static_branch and st_cols:
                            static_mat = (
                                X_all.reindex(index=np.array(pids, dtype=str))
                                .loc[:, st_cols]
                                .to_numpy(dtype=np.float32, copy=True)
                            )
                            logits = model_nn(X_seq, lens_t, torch.from_numpy(static_mat)).squeeze(-1).detach().cpu().numpy()
                        else:
                            logits = model_nn(X_seq, lens_t).squeeze(-1).detach().cpu().numpy()
                    if task_kind == "binary":
                        y_pred_all = sigmoid(logits)
                    else:
                        y_pred_all = logits

                    m_flat = M_seq.detach().cpu().numpy().reshape(-1).astype(bool) if hasattr(M_seq, "detach") else M_seq.reshape(-1).astype(bool)
                    y_pred = y_pred_all.reshape(-1)[m_flat]

            y_true = np.asarray(y_test, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[finite]
            y_pred = y_pred[finite]
            if y_true.size == 0:
                continue

            if task_kind == "binary":
                metrics = binary_metrics(y_true, y_pred).metrics
            else:
                metrics = regression_metrics(y_true, y_pred).metrics

            write_json(output / f"metrics_{model_name}_{split_name}.json", metrics)

            preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            preds_df.to_parquet(output / f"preds_{model_name}_{split_name}.parquet", index=False)

            rows.append({"model": model_name, "split": split_name, **metrics})

    write_json(output / "test_summary.json", {"records": rows})
    if rows:
        pd.DataFrame(rows).to_csv(output / "test_summary.csv", index=False)
