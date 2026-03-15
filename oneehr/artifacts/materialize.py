from __future__ import annotations

from pathlib import Path

import pandas as pd

from oneehr.artifacts.run_manifest import write_run_manifest
from oneehr.config.schema import ExperimentConfig
from oneehr.data.binning import bin_events
from oneehr.data.labels import normalize_patient_labels, normalize_time_labels, run_label_fn
from oneehr.data.static_postprocess import fit_transform_static_features
from oneehr.data.tabular import make_patient_tabular, make_time_tabular
from oneehr.utils.io import ensure_dir


def materialize_preprocess_artifacts(
    *,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg: ExperimentConfig,
    out_root: Path,
) -> None:
    """Materialize run artifacts used by train/test/hpo.

    Output conventions (MVP, schema_version=2):
    - Parquet for tables/matrices: `*.parquet`
    - JSON for schemas/metadata: `*.json`

    Writes:
    - binned.parquet
    - features/static/static_all.parquet (if enabled)
    - views/patient_tabular.parquet or views/time_tabular.parquet
    - run_manifest.json
    - labels.parquet (if labels are available)
    """

    out_root = ensure_dir(out_root)

    binned_df = None
    feat_cols: list[str] = []
    if dynamic is not None:
        if cfg.dataset.dynamic is None:
            raise ValueError("dataset.dynamic config is required when a dynamic table is provided.")
        binned = bin_events(dynamic, cfg.dataset.dynamic, cfg.preprocess)
        # Standard binned column order: keys -> label -> features
        binned_df = binned.table.copy()
        base_cols = [c for c in ["patient_id", "bin_time", "label"] if c in binned_df.columns]
        feat_cols = [c for c in binned_df.columns if c.startswith("num__") or c.startswith("cat__")]
        other_cols = [c for c in binned_df.columns if c not in set(base_cols + feat_cols)]
        binned_df = binned_df[base_cols + other_cols + feat_cols]
    else:
        # Static-only runs: create an empty binned table placeholder.
        binned_df = pd.DataFrame(columns=["patient_id", "bin_time", "label"])

    (out_root / "binned.parquet").write_bytes(binned_df.to_parquet(index=False))

    # Views (tabular)
    ensure_dir(out_root / "views")
    pt_path = None
    tm_path = None
    if cfg.task.prediction_mode == "patient":
        if feat_cols:
            Xp, yp = make_patient_tabular(binned_df)
            dfp = Xp.reset_index()
            # Do not force-drop unlabeled patients at preprocess time. Labels are
            # materialized separately (labels.parquet) and joined below.
            if yp is None or yp.empty:
                dfp["label"] = pd.NA
            else:
                dfp["label"] = yp.to_numpy()
            # Standard column order: keys -> label -> features
            dfp = dfp[["patient_id", "label", *feat_cols]]
        else:
            # Static-only: create a minimal patient view so downstream training
            # can still read patient_id/label and join static features.
            if static is None or static.empty:
                dfp = pd.DataFrame(columns=["patient_id", "label"])
            else:
                dfp = pd.DataFrame({"patient_id": static["patient_id"].astype(str).unique()})
                dfp["label"] = pd.NA
            dfp = dfp[["patient_id", "label"]]
        (out_root / "views" / "patient_tabular.parquet").write_bytes(dfp.to_parquet(index=False))
        pt_path = "views/patient_tabular.parquet"
    elif cfg.task.prediction_mode == "time":
        if not feat_cols:
            raise ValueError("prediction_mode='time' requires dynamic features; static-only is unsupported.")
        Xt, yt, key = make_time_tabular(binned_df)
        dft = key.copy().reset_index(drop=True)
        dft["label"] = yt.to_numpy()
        for c in feat_cols:
            dft[c] = Xt[c].to_numpy()
        dft = dft[["patient_id", "bin_time", "label", *feat_cols]]
        (out_root / "views" / "time_tabular.parquet").write_bytes(dft.to_parquet(index=False))
        tm_path = "views/time_tabular.parquet"
    else:
        raise ValueError(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")

    # Static (only from static.csv)
    static_raw = None
    if static is not None:
        static_raw = static
        # Normalize patient_id name to match pipeline expectations.
        # static.csv schema is fixed: must include patient_id.
    static_feat_cols: list[str] = []
    static_post_pipeline = None
    if static_raw is not None and not static_raw.empty:
        if "patient_id" not in static_raw.columns:
            raise ValueError("static.csv missing required column: patient_id")
        # patient_id is a join key, not a model feature.
        # Important: ensure we keep it out of the feature encoder even if the
        # column name casing differs (some datasets may use PatientID/patientID).
        id_like_cols = [c for c in static_raw.columns if str(c).lower() in {"patient_id", "patientid"}]
        static_raw = static_raw.drop(columns=id_like_cols, errors="ignore")
        static_all, _, _, static_art = fit_transform_static_features(
            raw_train=static_raw,
            raw_val=None,
            raw_test=None,
            pipeline=cfg.preprocess.pipeline,
        )
        static_feat_cols = list(static_all.columns)
        static_post_pipeline = None if static_art.fitted_postprocess is None else static_art.fitted_postprocess.pipeline
        ensure_dir(out_root / "features" / "static")
        (out_root / "features" / "static" / "static_all.parquet").write_bytes(static_all.to_parquet(index=True))

    write_run_manifest(
        out_root=out_root,
        cfg=cfg,
        dynamic_feature_columns=feat_cols,
        static_feature_columns=static_feat_cols,
        static_postprocess_pipeline=static_post_pipeline,
        patient_tabular_path=pt_path,
        time_tabular_path=tm_path,
    )

    labels_res = None if dynamic is None else run_label_fn(dynamic, static, label, cfg)
    if labels_res is not None:
        if cfg.task.prediction_mode == "patient":
            labels = normalize_patient_labels(labels_res.df)
        elif cfg.task.prediction_mode == "time":
            labels = normalize_time_labels(labels_res.df, cfg)
        else:
            raise ValueError(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")
        # Ensure stable labels schema.
        if cfg.task.prediction_mode == "patient":
            labels = labels[["patient_id", "label"]].copy()
        else:
            cols = ["patient_id", "bin_time", "label"]
            if "mask" in labels.columns:
                cols.append("mask")
            labels = labels[cols].copy()
        (out_root / "labels.parquet").write_bytes(labels.to_parquet(index=False))

        # Join labels onto the materialized tabular views so training can load
        # features+labels directly from `views/*.parquet`.
        if cfg.task.prediction_mode == "patient" and pt_path is not None:
            dfp = pd.read_parquet(out_root / pt_path)
            if "patient_id" in dfp.columns:
                dfp["patient_id"] = dfp["patient_id"].astype(str)
            labels_j = labels.copy()
            labels_j["patient_id"] = labels_j["patient_id"].astype(str)
            # If a label is missing for a patient, keep it as NaN (train will drop).
            dfp = dfp.drop(columns=["label"], errors="ignore").merge(labels_j, on="patient_id", how="left")
            dfp = dfp[["patient_id", "label", *feat_cols]]
            (out_root / pt_path).write_bytes(dfp.to_parquet(index=False))
        elif cfg.task.prediction_mode == "time" and tm_path is not None:
            dft = pd.read_parquet(out_root / tm_path)
            dft["patient_id"] = dft["patient_id"].astype(str)
            labels_j = labels.copy()
            labels_j["patient_id"] = labels_j["patient_id"].astype(str)
            labels_j["bin_time"] = pd.to_datetime(labels_j["bin_time"], errors="raise")
            # If a label is missing for a (patient,bin_time), keep as NaN (train will drop).
            dft = dft.drop(columns=["label"], errors="ignore").merge(
                labels_j[["patient_id", "bin_time", "label"]], on=["patient_id", "bin_time"], how="left"
            )
            dft = dft[["patient_id", "bin_time", "label", *feat_cols]]
            (out_root / tm_path).write_bytes(dft.to_parquet(index=False))
