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
    import pandas as pd

    from oneehr.config.load import load_experiment_config
    from oneehr.data.io import load_dynamic_table, load_label_table, load_static_table
    from oneehr.artifacts.read import read_run_manifest
    from oneehr.artifacts.inference import materialize_test_views
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.models.tabular import load_tabular_model, predict_tabular
    from oneehr.models.registry import build_model

    cfg0 = load_experiment_config(cfg_path)
    if run_dir is not None:
        run_root = Path(run_dir)
    else:
        run_root = cfg0.output.root / cfg0.output.run_name
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}")

    manifest = read_run_manifest(run_root)
    if manifest is None:
        raise SystemExit(f"Missing run_manifest.json under {run_root}. Run `oneehr preprocess` first.")

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

    TABULAR_MODELS = {"xgboost", "catboost", "rf", "dt", "gbdt"}
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
                    continue
                from oneehr.config.schema import TaskConfig

                task_cfg = TaskConfig(kind=task_kind, prediction_mode=mode)
                art = load_tabular_model(split_dir, task=task_cfg, kind=model_name)
                from oneehr.config.schema import TaskConfig

                task_cfg = TaskConfig(kind=task_kind, prediction_mode=mode)
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
                model_supports_static_branch = hasattr(model, "static_dim") and int(getattr(model, "static_dim", 0)) > 0

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
