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

    # Load preprocessed data
    binned = pd.read_parquet(run_dir / "preprocess" / "binned.parquet")
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
        preds_df = pd.DataFrame(columns=[
            "system", "patient_id", "y_true", "y_pred",
        ])
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
    from oneehr.training.trainer import sigmoid

    model, meta = load_checkpoint(model_dir)

    # Filter binned to test patients
    binned_test = binned[binned["patient_id"].astype(str).isin(test_pids)].copy()

    # Build y_true map
    y_true_map: dict[str, float] = {}  # patient-level
    y_true_time_map: dict[tuple[str, str], float] = {}  # (patient_id, bin_time) -> label
    if labels_df is not None:
        for _, row in labels_df.iterrows():
            pid = str(row["patient_id"])
            if pid in test_pids:
                if "bin_time" in labels_df.columns:
                    bt = str(row["bin_time"])
                    y_true_time_map[(pid, bt)] = float(row["label"])
                else:
                    y_true_map[pid] = float(row["label"])

    rows: list[dict] = []

    if isinstance(model, torch.nn.Module):
        # DL model
        model.eval()
        from oneehr.data.sequence import build_patient_sequences, pad_sequences
        from oneehr.data.tabular import has_static_branch

        # Load static features for models with a static branch
        run_dir = model_dir.parent.parent
        static_path = run_dir / "preprocess" / "static.parquet"
        static_tensor = None
        if has_static_branch(model) and static_path.exists():
            static_df = pd.read_parquet(static_path)
            if "patient_id" in static_df.columns:
                static_df = static_df.set_index("patient_id")
            static_df.index = static_df.index.astype(str)

        # Prepare PRISM extra inputs if applicable.
        prism_extra_kw: dict = {}
        if meta.get("model_name") == "prism":
            from oneehr.models.prism import build_time_delta_tensor

            obs_rates_list = meta.get("extra", {}).get("obs_rates")
            if obs_rates_list is not None:
                prism_extra_kw["obs_rates"] = torch.tensor(obs_rates_list, dtype=torch.float32)

            obs_mask_path = run_dir / "preprocess" / "obs_mask.parquet"
            schema_path = run_dir / "preprocess" / "feature_schema.json"
            if obs_mask_path.exists() and schema_path.exists():
                obs_mask_df = pd.read_parquet(obs_mask_path)
                # Build time_delta for test patients.
                test_mask = obs_mask_df[obs_mask_df["patient_id"].astype(str).isin(test_pids)]
                mask_cols = [c for c in feat_cols if c in obs_mask_df.columns]
                from oneehr.utils import parse_bin_size
                freq = parse_bin_size("1d")  # default; actual bin_size read below
                td_unit = pd.Timedelta(freq)

                td_map: dict[str, np.ndarray] = {}
                for pid, grp in test_mask.groupby("patient_id", sort=False):
                    grp = grp.sort_values("bin_time")
                    mv = grp[mask_cols].values.astype(np.float32)
                    T_p, D_p = mv.shape
                    td = np.full((T_p, D_p), 2.0, dtype=np.float32)
                    last_t = np.full(D_p, -np.inf)
                    times = grp["bin_time"].values
                    for t_i in range(T_p):
                        for d_i in range(D_p):
                            if mv[t_i, d_i] > 0.5:
                                if last_t[d_i] != -np.inf:
                                    gap = (pd.Timestamp(times[t_i]) - pd.Timestamp(times[int(last_t[d_i])])) / td_unit
                                    td[t_i, d_i] = min(float(gap), 2.0)
                                else:
                                    td[t_i, d_i] = 0.0
                                last_t[d_i] = t_i
                    td_map[str(pid)] = td
                prism_extra_kw["_td_map"] = td_map  # temporary; built into tensor below

        if mode == "patient":
            pids, seqs, lens = build_patient_sequences(binned_test, feat_cols)
            X_seq = pad_sequences(seqs, lens)
            lens_t = torch.from_numpy(lens)

            # Build static tensor aligned with pids
            if has_static_branch(model) and static_path.exists():
                s_vals = static_df.reindex(pids).fillna(0.0).to_numpy(dtype=np.float32)
                static_tensor = torch.from_numpy(s_vals)

            # Build PRISM time_delta tensor for test patients.
            extra_kw: dict = {}
            if "obs_rates" in prism_extra_kw:
                extra_kw["obs_rates"] = prism_extra_kw["obs_rates"]
            td_map = prism_extra_kw.pop("_td_map", None)
            if td_map is not None:
                from oneehr.models.prism import build_time_delta_tensor
                max_len = int(lens.max())
                extra_kw["time_delta"] = build_time_delta_tensor(td_map, list(pids), max_len, len(feat_cols))

            with torch.no_grad():
                if static_tensor is not None:
                    logits = model(X_seq, lens_t, static_tensor, **extra_kw).squeeze(-1).detach().cpu().numpy()
                else:
                    logits = model(X_seq, lens_t, **extra_kw).squeeze(-1).detach().cpu().numpy()

            if task_kind == "binary":
                y_pred_all = sigmoid(logits)
            else:
                y_pred_all = logits

            for pid, yp in zip(pids, y_pred_all.tolist()):
                rows.append({
                    "system": model_name,
                    "patient_id": str(pid),
                    "y_true": y_true_map.get(str(pid), float("nan")),
                    "y_pred": float(yp),
                })
        else:
            # Time mode DL
            from oneehr.data.sequence import build_time_sequences, pad_sequences

            if labels_df is None:
                return rows
            pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
                binned_test, labels_df, feat_cols,
            )
            X_seq = pad_sequences(seqs, lens)
            lens_t = torch.from_numpy(lens)

            # Build static tensor for time-mode if needed
            time_static = None
            if has_static_branch(model) and static_path.exists():
                s_vals = static_df.reindex(pids).fillna(0.0).to_numpy(dtype=np.float32)
                time_static = torch.from_numpy(s_vals)

            # Build PRISM extra for time-mode.
            time_extra_kw: dict = {}
            if "obs_rates" in prism_extra_kw:
                time_extra_kw["obs_rates"] = prism_extra_kw["obs_rates"]
            td_map_t = prism_extra_kw.pop("_td_map", None)
            if td_map_t is not None:
                from oneehr.models.prism import build_time_delta_tensor
                max_len_t = int(lens.max())
                time_extra_kw["time_delta"] = build_time_delta_tensor(td_map_t, list(pids), max_len_t, len(feat_cols))

            with torch.no_grad():
                if time_static is not None:
                    logits = model(X_seq, lens_t, time_static, **time_extra_kw).squeeze(-1).detach().cpu().numpy()
                else:
                    logits = model(X_seq, lens_t, **time_extra_kw).squeeze(-1).detach().cpu().numpy()

            for i, (pid, l) in enumerate(zip(pids, lens)):
                for t in range(l):
                    val = logits[i, t] if logits.ndim > 1 else logits[i]
                    yp = float(sigmoid(val)) if task_kind == "binary" else float(val)
                    bt = str(time_seqs[i][t])
                    rows.append({
                        "system": model_name,
                        "patient_id": str(pid),
                        "y_true": y_true_time_map.get((str(pid), bt), float("nan")),
                        "y_pred": yp,
                    })
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
            last = (
                binned_test.sort_values(["patient_id", "bin_time"], kind="stable")
                .groupby("patient_id", sort=False)[feat_cols]
                .last()
            )
            last.index = last.index.astype(str)
            last = _join_static(last)

            try:
                if task_kind == "binary":
                    y_pred = model.predict_proba(last[stored_feat_cols])[:, 1]
                else:
                    y_pred = model.predict(last[stored_feat_cols])
            except Exception:
                y_pred = model.predict(last[stored_feat_cols])

            for pid, yp in zip(last.index.tolist(), y_pred.tolist()):
                rows.append({
                    "system": model_name,
                    "patient_id": str(pid),
                    "y_true": y_true_map.get(str(pid), float("nan")),
                    "y_pred": float(yp),
                })
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
                if task_kind == "binary":
                    y_pred = model.predict_proba(X_test[stored_feat_cols])[:, 1]
                else:
                    y_pred = model.predict(X_test[stored_feat_cols])
            except Exception:
                y_pred = model.predict(X_test[stored_feat_cols])

            for i, yp in enumerate(y_pred.tolist()):
                pid = str(key.iloc[i]["patient_id"])
                bt = str(key.iloc[i]["bin_time"])
                rows.append({
                    "system": model_name,
                    "patient_id": pid,
                    "y_true": y_true_time_map.get((pid, bt), float("nan")),
                    "y_pred": float(yp),
                })

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
    from oneehr.eval.metrics import binary_metrics, regression_metrics

    if not rows:
        return {"task": {"kind": task_kind, "prediction_mode": mode}, "systems": []}

    df = pd.DataFrame(rows)
    system_results = []

    for system_name in df["system"].unique():
        sdf = df[df["system"] == system_name].copy()
        y_true = sdf["y_true"].to_numpy(dtype=float)
        y_pred = sdf["y_pred"].to_numpy(dtype=float)

        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[finite]
        y_pred = y_pred[finite]

        if y_true.size == 0:
            system_results.append({
                "name": system_name,
                "kind": "trained_model",
                "n": 0,
                "metrics": {},
            })
            continue

        if task_kind == "binary":
            metrics = binary_metrics(y_true, y_pred).metrics
        else:
            metrics = regression_metrics(y_true, y_pred).metrics

        # Determine kind
        kind = "trained_model"
        for sc in systems_cfg:
            if sc.name == system_name:
                kind = sc.kind
                break

        system_results.append({
            "name": system_name,
            "kind": kind,
            "n": int(y_true.size),
            "metrics": metrics,
        })

    return {
        "task": {"kind": task_kind, "prediction_mode": mode},
        "systems": system_results,
    }
