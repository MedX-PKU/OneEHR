"""Preprocessing artifact materialization.

Output structure:
    {run_dir}/preprocess/
        binned.parquet
        labels.parquet
        split.json
        static.parquet  (optional)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from oneehr.artifacts.manifest import write_manifest
from oneehr.config.schema import ExperimentConfig
from oneehr.data.binning import bin_events
from oneehr.data.labels import normalize_patient_labels, normalize_time_labels
from oneehr.data.splits import load_split, make_patient_index, make_split, save_split
from oneehr.data.tabular import fit_pipeline, fit_transform_static_features
from oneehr.utils import ensure_dir


def materialize_preprocess_artifacts(
    *,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg: ExperimentConfig,
    run_dir: Path,
) -> None:
    run_dir = ensure_dir(run_dir)
    pp_dir = ensure_dir(run_dir / "preprocess")

    # --- Binning ---
    feat_cols: list[str] = []
    if dynamic is not None:
        binned_res = bin_events(dynamic, None, cfg.preprocess)
        binned_df = binned_res.table.copy()
        feat_cols = [c for c in binned_df.columns if c.startswith("num__") or c.startswith("cat__")]
        base_cols = [c for c in ["patient_id", "bin_time", "label"] if c in binned_df.columns]
        other_cols = [c for c in binned_df.columns if c not in set(base_cols + feat_cols)]
        binned_df = binned_df[base_cols + other_cols + feat_cols]
    else:
        binned_df = pd.DataFrame(columns=["patient_id", "bin_time", "label"])

    binned_df.to_parquet(pp_dir / "binned.parquet", index=False)

    # Save feature schema and observation mask from binning.
    if dynamic is not None:
        (pp_dir / "feature_schema.json").write_text(
            json.dumps(binned_res.feature_schema, indent=2),
            encoding="utf-8",
        )
        binned_res.obs_mask.to_parquet(pp_dir / "obs_mask.parquet", index=False)

    # --- Labels ---
    if label is not None and not label.empty:
        # Convert raw label table format (label_value) to normalized (label)
        lab = label.copy()
        if "label" not in lab.columns and "label_value" in lab.columns:
            lab["label"] = lab["label_value"]
        if cfg.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(lab)
        else:
            labels_df = normalize_time_labels(lab, cfg)
        labels_df.to_parquet(pp_dir / "labels.parquet", index=False)

    # --- Split ---
    if dynamic is not None:
        patient_index = make_patient_index(dynamic)
    elif static is not None:
        pids = static["patient_id"].astype(str).dropna().unique()
        patient_index = pd.DataFrame({"patient_id": pids, "min_time": pd.NaT, "max_time": pd.NaT})
    else:
        raise SystemExit("dataset.dynamic or dataset.static is required.")

    split = make_split(patient_index, cfg.split)
    save_split(split, pp_dir / "split.json")

    # --- Fit preprocessing pipeline on train split ---
    if dynamic is not None:
        train_pids = set(load_split(pp_dir / "split.json").train.tolist())
        train_mask = binned_df["patient_id"].astype(str).isin(train_pids)
        X_train_for_pipeline = binned_df.loc[train_mask].copy()
        fitted = fit_pipeline(X_train_for_pipeline, cfg.preprocess.pipeline)
        torch.save(fitted, pp_dir / "fitted_pipeline.pt")

    # --- Static features ---
    static_feat_cols: list[str] | None = None
    if static is not None and not static.empty:
        if "patient_id" not in static.columns:
            raise ValueError("static.csv missing required column: patient_id")
        id_like = [c for c in static.columns if str(c).lower() in {"patient_id", "patientid"}]
        static_raw = static.drop(columns=id_like, errors="ignore")
        # Filter pipeline ops that require time ordering (forward_fill)
        # since static features are patient-level only
        _TIME_OPS = {"forward_fill"}
        static_pipeline = [s for s in cfg.preprocess.pipeline if s.get("op") not in _TIME_OPS]
        static_all, _, _, static_art = fit_transform_static_features(
            raw_train=static_raw,
            raw_val=None,
            raw_test=None,
            pipeline=static_pipeline,
        )
        static_feat_cols = list(static_all.columns)
        # Set patient_id as index before saving
        static_all.index = static["patient_id"].astype(str).values[: len(static_all)]
        static_all.index.name = "patient_id"
        static_all.to_parquet(pp_dir / "static.parquet", index=True)

    # --- Manifest ---
    write_manifest(
        out_dir=run_dir,
        cfg=cfg,
        feature_columns=feat_cols,
        static_feature_columns=static_feat_cols,
    )
