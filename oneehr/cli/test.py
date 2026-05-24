"""oneehr test subcommand.

Produces:
    {run_dir}/test/predictions.parquet  — all systems × test patients
    {run_dir}/test/metrics.json         — per-system metrics
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oneehr.utils import ensure_dir, write_json


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _prob_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    def _key(col: str):
        suffix = col.rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, col)

    return sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=_key,
    )


def _prediction_payload(raw_pred, task_kind: str, *, is_probability: bool = False) -> dict:
    arr = np.asarray(raw_pred, dtype=float)
    if task_kind == "binary":
        score = arr.reshape(-1)[0] if is_probability else _sigmoid(arr).reshape(-1)[0]
        return {"y_pred": float(score)}
    if task_kind == "multiclass":
        if arr.ndim == 0:
            return {"y_pred": float(arr)}
        probs = arr if is_probability else _softmax(arr)
        payload = {"y_pred": float(int(probs.argmax()))}
        payload.update({f"y_prob_{i}": float(p) for i, p in enumerate(probs.reshape(-1).tolist())})
        return payload
    if task_kind == "multilabel":
        probs = _sigmoid(arr).reshape(-1)
        payload = {"y_pred": float(int((probs >= 0.5).sum()))}
        payload.update({f"y_prob_{i}": float(p) for i, p in enumerate(probs.tolist())})
        return payload
    return {"y_pred": float(arr.reshape(-1)[0])}


def _label_matrix_columns(labels_df: pd.DataFrame) -> list[str]:
    return [c for c in labels_df.columns if c not in {"patient_id", "bin_time", "label_time", "label", "mask"}]


def _truth_payload(raw_true, task_kind: str) -> dict:
    if task_kind == "multilabel":
        arr = np.asarray(raw_true, dtype=float)
        if arr.ndim > 0 and arr.size > 1:
            payload = {"y_true": float("nan")}
            payload.update({f"y_true_{i}": float(v) for i, v in enumerate(arr.reshape(-1).tolist())})
            return payload
    return {"y_true": float(raw_true)}


def _apply_pipeline(run_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Load fitted pipeline and apply to dataframe, then fill residual NaN."""
    pipeline_path = run_dir / "preprocess" / "fitted_pipeline.pt"
    if not pipeline_path.exists():
        for col in df.columns:
            if col.startswith("num__"):
                df[col] = df[col].fillna(0.0)
        return df

    from oneehr.data.tabular import transform_pipeline

    fitted = torch.load(pipeline_path, weights_only=False)
    df = transform_pipeline(df, fitted)
    for col in df.columns:
        if col.startswith("num__"):
            df[col] = df[col].fillna(0.0)
    return df


def run_test(cfg_path: str, force: bool) -> None:
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(cfg_path)
    run_dir = cfg.run_dir()
    test_dir = run_dir / "test"

    if test_dir.exists() and not force:
        raise SystemExit(f"Test artifacts exist at {test_dir}. Use --force to overwrite.")
    if test_dir.exists() and force:
        shutil.rmtree(test_dir)
    ensure_dir(test_dir)

    from oneehr.artifacts.manifest import read_manifest
    from oneehr.data.splits import load_split

    manifest = read_manifest(run_dir)
    feat_cols = manifest["feature_columns"]
    split = load_split(run_dir / "preprocess" / "split.json")
    test_pids = set(split.test.tolist())

    if not test_pids:
        raise SystemExit("No test patients in split.")

    # Load preprocessed data and apply pipeline
    binned = pd.read_parquet(run_dir / "preprocess" / "binned.parquet")
    binned = _apply_pipeline(run_dir, binned)
    labels_path = run_dir / "preprocess" / "labels.parquet"
    labels_df = pd.read_parquet(labels_path) if labels_path.exists() else None

    task_kind = cfg.task.kind
    mode = cfg.task.prediction_mode

    all_rows: list[dict] = []

    # --- Trained models ---
    train_dir = run_dir / "train"
    if train_dir.exists():
        for model_dir in sorted(train_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            ckpt_path = model_dir / "checkpoint.ckpt"
            if not ckpt_path.exists():
                continue

            print(f"Testing trained model: {model_name}")
            rows = _predict_trained_model(
                model_dir=model_dir,
                model_name=model_name,
                binned=binned,
                labels_df=labels_df,
                feat_cols=feat_cols,
                test_pids=test_pids,
                task_kind=task_kind,
                mode=mode,
            )
            all_rows.extend(rows)

    # --- LLM/Agent systems ---
    for system_cfg in cfg.systems:
        print(f"Testing system: {system_cfg.name}")
        rows = _predict_llm_system(
            system_cfg=system_cfg,
            binned=binned,
            labels_df=labels_df,
            feat_cols=feat_cols,
            test_pids=test_pids,
            task_kind=task_kind,
            mode=mode,
        )
        all_rows.extend(rows)

    # Build predictions.parquet
    if all_rows:
        preds_df = pd.DataFrame(all_rows)
        preds_df.to_parquet(test_dir / "predictions.parquet", index=False)
    else:
        # Empty predictions
        preds_df = pd.DataFrame(
            columns=[
                "system",
                "patient_id",
                "y_true",
                "y_pred",
            ]
        )
        preds_df.to_parquet(test_dir / "predictions.parquet", index=False)

    # Build metrics.json
    metrics = _compute_metrics(all_rows, task_kind, mode, cfg.systems)
    write_json(test_dir / "metrics.json", metrics)

    print(f"Test results written to {test_dir}")


def _predict_trained_model(
    *,
    model_dir: Path,
    model_name: str,
    binned: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    feat_cols: list[str],
    test_pids: set[str],
    task_kind: str,
    mode: str,
) -> list[dict]:
    """Load a checkpoint and produce prediction rows for test patients."""
    from oneehr.training.persistence import load_checkpoint

    model, meta = load_checkpoint(model_dir)

    # Filter binned to test patients
    binned_test = binned[binned["patient_id"].astype(str).isin(test_pids)].copy()

    # Build y_true map
    y_true_map: dict[str, object] = {}  # patient-level
    y_true_time_map: dict[tuple[str, str], object] = {}  # (patient_id, bin_time) -> label
    if labels_df is not None:
        label_cols = _label_matrix_columns(labels_df) if task_kind == "multilabel" else []
        for _, row in labels_df.iterrows():
            pid = str(row["patient_id"])
            if pid in test_pids:
                if "bin_time" in labels_df.columns:
                    bt = str(row["bin_time"])
                    y_true_time_map[(pid, bt)] = row[label_cols].to_numpy(dtype=float) if label_cols else float(row["label"])
                else:
                    y_true_map[pid] = row[label_cols].to_numpy(dtype=float) if label_cols else float(row["label"])

    rows: list[dict] = []

    if isinstance(model, torch.nn.Module):
        # DL model
        model.eval()
        from oneehr.data.sequence import build_patient_sequences, pad_sequences
        from oneehr.data.tabular import has_static_branch
        from oneehr.models.runtime import build_inference_extra

        # Load static features for models with a static branch
        run_dir = model_dir.parent.parent
        static_path = run_dir / "preprocess" / "static.parquet"
        static_tensor = None
        if has_static_branch(model) and static_path.exists():
            static_df = pd.read_parquet(static_path)
            if "patient_id" in static_df.columns:
                static_df = static_df.set_index("patient_id")
            static_df.index = static_df.index.astype(str)

        if mode == "patient":
            pids, seqs, lens = build_patient_sequences(binned_test, feat_cols)
            X_seq = pad_sequences(seqs, lens)
            lens_t = torch.from_numpy(lens)

            # Build static tensor aligned with pids
            if has_static_branch(model) and static_path.exists():
                s_vals = static_df.reindex(pids).fillna(0.0).to_numpy(dtype=np.float32)
                static_tensor = torch.from_numpy(s_vals)
            extra_kw = build_inference_extra(
                model_name=meta.get("model_name", model_name),
                meta=meta,
                run_dir=run_dir,
                feat_cols=feat_cols,
                patient_ids=list(pids),
                max_len=int(lens.max()) if len(lens) else 0,
            )
            extra_kw = {k: v for k, v in extra_kw.items() if not k.startswith("_")}

            with torch.no_grad():
                if static_tensor is not None:
                    logits = model(X_seq, lens_t, static_tensor, **extra_kw).squeeze(-1).detach().cpu().numpy()
                else:
                    logits = model(X_seq, lens_t, **extra_kw).squeeze(-1).detach().cpu().numpy()

            for pid, raw in zip(pids, logits.tolist()):
                row = {
                    "system": model_name,
                    "patient_id": str(pid),
                }
                row.update(_truth_payload(y_true_map.get(str(pid), float("nan")), task_kind))
                row.update(_prediction_payload(raw, task_kind))
                rows.append(row)
        else:
            # Time mode DL
            from oneehr.data.sequence import build_time_sequences, pad_sequences

            if labels_df is None:
                return rows
            pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
                binned_test,
                labels_df,
                feat_cols,
            )
            X_seq = pad_sequences(seqs, lens)
            lens_t = torch.from_numpy(lens)

            # Build static tensor for time-mode if needed
            time_static = None
            if has_static_branch(model) and static_path.exists():
                s_vals = static_df.reindex(pids).fillna(0.0).to_numpy(dtype=np.float32)
                time_static = torch.from_numpy(s_vals)
            time_extra_kw = build_inference_extra(
                model_name=meta.get("model_name", model_name),
                meta=meta,
                run_dir=run_dir,
                feat_cols=feat_cols,
                patient_ids=list(pids),
                max_len=int(lens.max()) if len(lens) else 0,
            )
            time_extra_kw = {k: v for k, v in time_extra_kw.items() if not k.startswith("_")}

            with torch.no_grad():
                if time_static is not None:
                    logits = model(X_seq, lens_t, time_static, **time_extra_kw).squeeze(-1).detach().cpu().numpy()
                else:
                    logits = model(X_seq, lens_t, **time_extra_kw).squeeze(-1).detach().cpu().numpy()

            for i, (pid, seq_len) in enumerate(zip(pids, lens)):
                for t in range(seq_len):
                    val = logits[i, t] if logits.ndim > 1 else logits[i]
                    bt = str(time_seqs[i][t])
                    row = {
                        "system": model_name,
                        "patient_id": str(pid),
                    }
                    row.update(_truth_payload(y_true_time_map.get((str(pid), bt), float("nan")), task_kind))
                    row.update(_prediction_payload(val, task_kind))
                    rows.append(row)
    else:
        # ML model (XGBoost, CatBoost etc.) — loaded via torch.save
        if binned_test.empty:
            return rows

        run_dir = model_dir.parent.parent
        static_path = run_dir / "preprocess" / "static.parquet"
        stored_feat_cols = meta.get("feature_columns", feat_cols)

        def _join_static(df: pd.DataFrame) -> pd.DataFrame:
            if static_path.exists():
                static_df = pd.read_parquet(static_path)
                if "patient_id" in static_df.columns:
                    static_df = static_df.set_index("patient_id")
                static_df.index = static_df.index.astype(str)
                overlap = [c for c in static_df.columns if c in df.columns]
                static_use = static_df.drop(columns=overlap, errors="ignore")
                df = df.join(static_use, how="left").fillna(0.0)
            return df

        if mode == "patient":
            last = binned_test.sort_values(["patient_id", "bin_time"], kind="stable").groupby("patient_id", sort=False)[feat_cols].last()
            last.index = last.index.astype(str)
            last = _join_static(last)

            try:
                pred_is_prob = False
                if task_kind == "binary":
                    y_pred = model.predict_proba(last[stored_feat_cols])[:, 1]
                    pred_is_prob = True
                elif task_kind == "multiclass" and hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(last[stored_feat_cols])
                    pred_is_prob = True
                else:
                    y_pred = model.predict(last[stored_feat_cols])
            except Exception:
                y_pred = model.predict(last[stored_feat_cols])
                pred_is_prob = False

            for pid, yp in zip(last.index.tolist(), y_pred.tolist()):
                row = {
                    "system": model_name,
                    "patient_id": str(pid),
                }
                row.update(_truth_payload(y_true_map.get(str(pid), float("nan")), task_kind))
                row.update(_prediction_payload(yp, task_kind, is_probability=pred_is_prob))
                rows.append(row)
        else:
            # Time-level ML prediction
            df = binned_test[["patient_id", "bin_time", *feat_cols]].copy()
            df["patient_id"] = df["patient_id"].astype(str)
            key = df[["patient_id", "bin_time"]].reset_index(drop=True)
            X_test = df[feat_cols].reset_index(drop=True)

            # Join static via patient_id index
            X_test.index = df["patient_id"].values
            X_test = _join_static(X_test)
            X_test = X_test.reset_index(drop=True)

            try:
                pred_is_prob = False
                if task_kind == "binary":
                    y_pred = model.predict_proba(X_test[stored_feat_cols])[:, 1]
                    pred_is_prob = True
                elif task_kind == "multiclass" and hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_test[stored_feat_cols])
                    pred_is_prob = True
                else:
                    y_pred = model.predict(X_test[stored_feat_cols])
            except Exception:
                y_pred = model.predict(X_test[stored_feat_cols])
                pred_is_prob = False

            for i, yp in enumerate(y_pred.tolist()):
                pid = str(key.iloc[i]["patient_id"])
                bt = str(key.iloc[i]["bin_time"])
                row = {
                    "system": model_name,
                    "patient_id": pid,
                }
                row.update(_truth_payload(y_true_time_map.get((pid, bt), float("nan")), task_kind))
                row.update(_prediction_payload(yp, task_kind, is_probability=pred_is_prob))
                rows.append(row)

    return rows


def _predict_llm_system(
    *,
    system_cfg,
    binned: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    feat_cols: list[str],
    test_pids: set[str],
    task_kind: str,
    mode: str,
) -> list[dict]:
    """Placeholder for LLM/agent system prediction.

    Full LLM integration requires the agent runtime module.
    Returns empty rows if the agent module is not available.
    """
    rows: list[dict] = []
    try:
        from oneehr.agent.runtime import run_system_on_patients

        rows = run_system_on_patients(
            system_cfg=system_cfg,
            binned=binned,
            labels_df=labels_df,
            feat_cols=feat_cols,
            test_pids=test_pids,
            task_kind=task_kind,
        )
    except ImportError:
        print(f"  Warning: agent runtime not available, skipping {system_cfg.name}")
    return rows


def _compute_metrics(
    rows: list[dict],
    task_kind: str,
    mode: str,
    systems_cfg: list,
) -> dict:
    """Compute per-system metrics from prediction rows."""
    from oneehr.eval.metrics import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics

    if not rows:
        return {"task": {"kind": task_kind, "prediction_mode": mode}, "systems": []}

    df = pd.DataFrame(rows)
    system_results = []

    for system_name in df["system"].unique():
        sdf = df[df["system"] == system_name].copy()

        if task_kind == "multilabel":
            true_cols = _prob_cols(sdf, "y_true_")
            prob_cols = _prob_cols(sdf, "y_prob_")
            if true_cols and prob_cols:
                n_labels = min(len(true_cols), len(prob_cols))
                y_true_ml = sdf[true_cols[:n_labels]].to_numpy(dtype=float)
                y_score_ml = sdf[prob_cols[:n_labels]].to_numpy(dtype=float)
                finite_ml = np.isfinite(y_true_ml).all(axis=1) & np.isfinite(y_score_ml).all(axis=1)
                y_true_ml = y_true_ml[finite_ml].astype(int)
                y_score_ml = y_score_ml[finite_ml]
                metrics = multilabel_metrics(y_true_ml, y_score_ml).metrics if y_true_ml.size else {}
                n_eval = int(y_true_ml.shape[0])
            else:
                metrics = {}
                n_eval = 0

            kind = "trained_model"
            for sc in systems_cfg:
                if sc.name == system_name:
                    kind = sc.kind
                    break
            system_results.append({"name": system_name, "kind": kind, "n": n_eval, "metrics": metrics})
            continue

        y_true = sdf["y_true"].to_numpy(dtype=float)
        y_pred = sdf["y_pred"].to_numpy(dtype=float)

        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[finite]
        y_pred = y_pred[finite]

        if y_true.size == 0:
            system_results.append(
                {
                    "name": system_name,
                    "kind": "trained_model",
                    "n": 0,
                    "metrics": {},
                }
            )
            continue

        if task_kind == "binary":
            metrics = binary_metrics(y_true, y_pred).metrics
        elif task_kind == "multiclass":
            # For multiclass, y_pred columns may be stored as separate prob cols
            prob_cols = _prob_cols(sdf, "y_prob_")
            if prob_cols:
                y_probs = sdf[prob_cols].to_numpy(dtype=float)[finite]
            else:
                y_probs = y_pred  # fallback: argmax-style
            num_classes = max(int(y_true.max()) + 1, len(prob_cols))
            metrics = multiclass_metrics(y_true.astype(int), y_probs, num_classes=num_classes).metrics
        else:
            metrics = regression_metrics(y_true, y_pred).metrics

        # Determine kind
        kind = "trained_model"
        for sc in systems_cfg:
            if sc.name == system_name:
                kind = sc.kind
                break

        system_results.append(
            {
                "name": system_name,
                "kind": kind,
                "n": int(y_true.size),
                "metrics": metrics,
            }
        )

    return {
        "task": {"kind": task_kind, "prediction_mode": mode},
        "systems": system_results,
    }
