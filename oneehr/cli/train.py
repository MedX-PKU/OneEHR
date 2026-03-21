"""oneehr train subcommand."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oneehr.models import TABULAR_MODELS, DL_MODELS
from oneehr.utils import ensure_dir, write_json


def run_train(cfg_path: str, force: bool) -> None:
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(cfg_path)
    run_dir = cfg.run_dir()

    if not (run_dir / "preprocess").exists():
        raise SystemExit(
            f"Preprocessed artifacts not found at {run_dir / 'preprocess'}. "
            "Run `oneehr preprocess` first."
        )

    train_dir = run_dir / "train"
    if train_dir.exists() and not force:
        raise SystemExit(f"Train artifacts exist at {train_dir}. Use --force to overwrite.")
    if train_dir.exists() and force:
        shutil.rmtree(train_dir)

    if not cfg.models:
        raise SystemExit("No [[models]] defined in config.")

    # Load preprocessed data
    from oneehr.artifacts.manifest import read_manifest
    from oneehr.data.splits import load_split

    manifest = read_manifest(run_dir)
    feat_cols = manifest["feature_columns"]
    split = load_split(run_dir / "preprocess" / "split.json")

    binned = pd.read_parquet(run_dir / "preprocess" / "binned.parquet")
    labels_path = run_dir / "preprocess" / "labels.parquet"
    labels_df = pd.read_parquet(labels_path) if labels_path.exists() else None

    # Build tabular view
    time_key: pd.DataFrame | None = None
    if cfg.task.prediction_mode == "patient":
        X, y = _build_patient_tabular(binned, labels_df, feat_cols)
    else:
        X, y, time_key = _build_time_tabular(binned, labels_df, feat_cols)

    # Load static features if available
    static_path = run_dir / "preprocess" / "static.parquet"
    static_all = pd.read_parquet(static_path) if static_path.exists() else None

    for model_cfg in cfg.models:
        model_name = model_cfg.name
        model_out = ensure_dir(train_dir / model_name)

        print(f"Training {model_name}...")

        if model_name in TABULAR_MODELS:
            _train_tabular(
                model_name=model_name,
                model_cfg=model_cfg,
                X=X, y=y, split=split,
                cfg=cfg, model_out=model_out,
                feat_cols=feat_cols,
                static_all=static_all,
                time_key=time_key,
            )
        elif model_name in DL_MODELS:
            _train_dl(
                model_name=model_name,
                model_cfg=model_cfg,
                binned=binned, labels_df=labels_df,
                X=X, y=y, split=split,
                cfg=cfg, model_out=model_out,
                feat_cols=feat_cols,
                static_all=static_all,
            )
        else:
            raise SystemExit(f"Unsupported model: {model_name!r}")

        print(f"  Saved to {model_out}")


def _build_patient_tabular(
    binned: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build patient-level tabular view: last observation per patient."""
    if binned.empty:
        X = pd.DataFrame(columns=feat_cols)
        X.index.name = "patient_id"
        return X, pd.Series(dtype=float, name="label")

    last = (
        binned.sort_values(["patient_id", "bin_time"], kind="stable")
        .groupby("patient_id", sort=False)[feat_cols]
        .last()
    )
    last.index = last.index.astype(str)
    last.index.name = "patient_id"

    if labels_df is not None and "patient_id" in labels_df.columns and "label" in labels_df.columns:
        lab = labels_df[["patient_id", "label"]].copy()
        lab["patient_id"] = lab["patient_id"].astype(str)
        lab = lab.drop_duplicates(subset=["patient_id"], keep="last").set_index("patient_id")
        last = last.join(lab, how="inner")
        y = last.pop("label")
    else:
        y = pd.Series(np.nan, index=last.index, name="label")

    return last, y


def _build_time_tabular(
    binned: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = binned[["patient_id", "bin_time", *feat_cols]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    key = df[["patient_id", "bin_time"]].reset_index(drop=True)
    X = df[feat_cols].reset_index(drop=True)

    if labels_df is not None:
        merged = key.merge(
            labels_df[["patient_id", "bin_time", "label"]],
            on=["patient_id", "bin_time"], how="left",
        )
        y = merged["label"].astype(float)
    else:
        y = pd.Series(np.nan, index=X.index, name="label")

    return X, y, key


def _train_tabular(
    *,
    model_name: str,
    model_cfg,
    X: pd.DataFrame,
    y: pd.Series,
    split,
    cfg,
    model_out: Path,
    feat_cols: list[str],
    static_all: pd.DataFrame | None,
    time_key: pd.DataFrame | None = None,
) -> None:
    from oneehr.models.tree import train_tabular_model, predict_tabular
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.training.persistence import save_checkpoint

    # Split data — for time-level, use the key's patient_id column
    if time_key is not None:
        pids = time_key["patient_id"].astype(str)
        train_mask = pids.isin(split.train).values
        val_mask = pids.isin(split.val).values
    else:
        train_mask = X.index.astype(str).isin(split.train)
        val_mask = X.index.astype(str).isin(split.val)

    X_train, y_train = X.loc[train_mask].copy(), y.loc[train_mask].to_numpy()
    X_val, y_val = X.loc[val_mask].copy(), y.loc[val_mask].to_numpy()

    # Drop NaN labels
    keep_tr = ~np.isnan(y_train.astype(float))
    keep_va = ~np.isnan(y_val.astype(float))
    X_train, y_train = X_train.iloc[keep_tr], y_train[keep_tr]
    X_val, y_val = X_val.iloc[keep_va], y_val[keep_va]

    # Join static features
    if static_all is not None:
        overlap = [c for c in static_all.columns if c in X_train.columns]
        static_use = static_all.drop(columns=overlap, errors="ignore")
        X_train = X_train.join(static_use, how="left").fillna(0.0)
        X_val = X_val.join(static_use, how="left").fillna(0.0)

    art = train_tabular_model(
        model_name=model_name,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        task=cfg.task,
        params=model_cfg.params,
        seed=cfg.trainer.seed,
    )

    # Compute val metrics
    y_val_pred = predict_tabular(art, X_val, cfg.task)
    if cfg.task.kind == "binary":
        metrics = binary_metrics(y_val.astype(float), y_val_pred.astype(float)).metrics
    else:
        metrics = regression_metrics(y_val.astype(float), y_val_pred.astype(float)).metrics

    save_checkpoint(
        out_dir=model_out,
        model=art.model,
        model_name=model_name,
        params=model_cfg.params,
        train_metrics=metrics,
        feature_columns=art.feature_columns,
    )


def _train_dl(
    *,
    model_name: str,
    model_cfg,
    binned: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    X: pd.DataFrame,
    y: pd.Series,
    split,
    cfg,
    model_out: Path,
    feat_cols: list[str],
    static_all: pd.DataFrame | None,
) -> None:
    from oneehr.models import build_dl_model
    from oneehr.training.trainer import fit_model
    from oneehr.training.persistence import save_checkpoint
    from oneehr.data.sequence import build_patient_sequences, build_time_sequences, pad_sequences
    from oneehr.data.tabular import has_static_branch

    # Auto-detect static_dim for models that support it
    _STATIC_MODELS = {"concare", "grasp", "mcgru", "dragent", "prism"}
    if (
        model_name in _STATIC_MODELS
        and static_all is not None
        and "static_dim" not in model_cfg.params
    ):
        model_cfg = type(model_cfg)(
            name=model_cfg.name,
            params={**model_cfg.params, "static_dim": static_all.shape[1]},
        )

    input_dim = len(feat_cols)

    # PRISM-specific: prepare extra inputs (dim_list, centers, obs_rates, time_delta).
    extra: dict | None = None
    extra_meta: dict = {}
    if model_name == "prism":
        import json as _json
        from oneehr.models.prism import prepare_prism_inputs, build_time_delta_tensor

        run_dir = model_out.parent.parent
        schema_path = run_dir / "preprocess" / "feature_schema.json"
        obs_mask_path = run_dir / "preprocess" / "obs_mask.parquet"

        feature_schema = _json.loads(schema_path.read_text(encoding="utf-8"))
        obs_mask_df = pd.read_parquet(obs_mask_path)

        train_pids = set(str(p) for p in split.train)
        prism_data = prepare_prism_inputs(
            binned, obs_mask_df, feat_cols, feature_schema, train_pids,
            n_clusters=int(model_cfg.params.get("n_clusters", 10)),
            bin_size=cfg.preprocess.bin_size,
        )
        # Inject dim_list and centers into model params.
        model_cfg = type(model_cfg)(
            name=model_cfg.name,
            params={**model_cfg.params, "dim_list": prism_data["dim_list"], "centers": prism_data["centers"]},
        )
        # Build padded time_delta tensor aligned with all patients.
        from oneehr.data.sequence import build_patient_sequences, pad_sequences
        all_pids, all_seqs, all_lens = build_patient_sequences(binned, feat_cols)
        max_len = int(all_lens.max())
        td_tensor = build_time_delta_tensor(prism_data["time_delta_map"], all_pids, max_len, input_dim)

        extra = {
            "obs_rates": prism_data["obs_rates"],  # [D] — broadcast, not batched
            "time_delta": td_tensor,               # [N, T, D] — batched
        }
        extra_meta = {"obs_rates": prism_data["obs_rates"].tolist()}

    model = build_dl_model(model_cfg, input_dim=input_dim, mode=cfg.task.prediction_mode)
    model_supports_static = has_static_branch(model)

    # Build sequences and train using the trainer
    if cfg.task.prediction_mode == "patient":
        y_map = {}
        if labels_df is not None:
            for _, row in labels_df.iterrows():
                y_map[str(row["patient_id"])] = float(row["label"])

        trained_model, train_metrics = fit_model(
            model=model, binned=binned, split=split,
            feat_cols=feat_cols, y_map=y_map,
            cfg=cfg.trainer, task=cfg.task,
            mode="patient",
            static=static_all if model_supports_static else None,
            extra=extra,
        )
    else:
        trained_model, train_metrics = fit_model(
            model=model, binned=binned, split=split,
            feat_cols=feat_cols, labels_df=labels_df,
            cfg=cfg.trainer, task=cfg.task,
            mode="time",
            static=static_all if model_supports_static else None,
            extra=extra,
        )

    # Build params for checkpoint (exclude non-serializable tensors).
    ckpt_params = {k: v for k, v in model_cfg.params.items() if not isinstance(v, torch.Tensor)}
    save_checkpoint(
        out_dir=model_out,
        model=trained_model,
        model_name=model_name,
        params=ckpt_params,
        train_metrics=train_metrics,
        feature_columns=feat_cols,
        extra_meta=extra_meta,
    )
