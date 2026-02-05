from __future__ import annotations

from pathlib import Path

import pandas as pd

from oneehr.artifacts.run_manifest import write_run_manifest
from oneehr.config.schema import ExperimentConfig
from oneehr.data.binning import bin_events
from oneehr.data.labels import normalize_patient_labels, normalize_time_labels, run_label_fn
from oneehr.data.static_features import build_static_features
from oneehr.data.static_postprocess import fit_transform_static_features
from oneehr.data.tabular import make_patient_tabular, make_time_tabular
from oneehr.utils.io import ensure_dir, write_json


def materialize_preprocess_artifacts(
    *,
    events: pd.DataFrame,
    cfg: ExperimentConfig,
    out_root: Path,
) -> None:
    """Materialize run artifacts used by train/test/hpo.

    Output conventions (MVP, schema_version=2):
    - Parquet for tables/matrices: `*.parquet`
    - JSON for schemas/metadata: `*.json`

    Writes:
    - binned.parquet
    - features/dynamic/feature_columns.json
    - features/static/feature_columns.json + features/static/static_all.parquet (if enabled)
    - views/patient_tabular.parquet or views/time_tabular.parquet
    - run_manifest.json
    - labels.parquet (if labels.fn provided)
    """

    out_root = ensure_dir(out_root)

    # If labels are generated via label_fn, attach them to events before binning.
    labels_res0 = run_label_fn(events, cfg)
    if labels_res0 is not None and cfg.task.prediction_mode == "patient":
        labels0 = normalize_patient_labels(labels_res0.df)
        ev2 = events.copy()
        pid_col = cfg.dataset.patient_id_col
        ev2[pid_col] = ev2[pid_col].astype(str)
        ev2 = ev2.merge(labels0, left_on=pid_col, right_on="patient_id", how="left")
        # Keep original pid column; drop helper column if it is not the configured pid column.
        if pid_col != "patient_id" and "patient_id" in ev2.columns:
            ev2 = ev2.drop(columns=["patient_id"])
        # Mirror label into dataset.label_col so downstream code sees it.
        if cfg.dataset.label_col != "label":
            ev2[cfg.dataset.label_col] = ev2["label"]
        events_for_binning = ev2
    elif labels_res0 is not None and cfg.task.prediction_mode == "time":
        # For time mode, we keep labels separate and rely on run.load_time_view.
        events_for_binning = events
    else:
        events_for_binning = events

    binned = bin_events(events_for_binning, cfg.dataset, cfg.preprocess)
    # Standard binned column order: keys -> label -> features
    binned_df = binned.table.copy()
    base_cols = [c for c in ["patient_id", "bin_time", "label"] if c in binned_df.columns]
    feat_cols = [c for c in binned_df.columns if c.startswith("num__") or c.startswith("cat__")]
    other_cols = [c for c in binned_df.columns if c not in set(base_cols + feat_cols)]
    binned_df = binned_df[base_cols + other_cols + feat_cols]
    (out_root / "binned.parquet").write_bytes(binned_df.to_parquet(index=False))

    # Dynamic feature space
    ensure_dir(out_root / "features" / "dynamic")
    write_json(out_root / "features" / "dynamic" / "feature_columns.json", {"feature_columns": feat_cols})

    # Views (tabular)
    ensure_dir(out_root / "views")
    pt_path = None
    tm_path = None
    if cfg.task.prediction_mode == "patient":
        Xp, yp = make_patient_tabular(binned_df)
        dfp = Xp.reset_index()
        dfp["label"] = yp.to_numpy()
        # Standard column order: keys -> label -> features
        dfp = dfp[["patient_id", "label", *feat_cols]]
        (out_root / "views" / "patient_tabular.parquet").write_bytes(dfp.to_parquet(index=False))
        pt_path = "views/patient_tabular.parquet"
    elif cfg.task.prediction_mode == "time":
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

    # Static
    static_raw = build_static_features(events, cfg.dataset, cfg.static_features)
    static_feat_cols: list[str] = []
    static_post_pipeline = None
    if static_raw is not None and not static_raw.empty and cfg.static_features.enabled:
        static_all, _, _, static_art = fit_transform_static_features(
            raw_train=static_raw,
            raw_val=None,
            raw_test=None,
            pipeline=cfg.preprocess.pipeline,
        )
        static_feat_cols = list(static_all.columns)
        static_post_pipeline = None if static_art.fitted_postprocess is None else static_art.fitted_postprocess.pipeline
        ensure_dir(out_root / "features" / "static")
        write_json(out_root / "features" / "static" / "feature_columns.json", {"feature_columns": static_feat_cols})
        (out_root / "features" / "static" / "static_all.parquet").write_bytes(static_all.to_parquet(index=True))

    write_run_manifest(
        out_root=out_root,
        cfg=cfg,
        dynamic_feature_columns=feat_cols,
        static_raw_cols=None if static_raw is None else list(static_raw.columns),
        static_feature_columns=static_feat_cols,
        static_postprocess_pipeline=static_post_pipeline,
        patient_tabular_path=pt_path,
        time_tabular_path=tm_path,
    )

    labels_res = labels_res0
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
    else:
        # Labels are optional. Persist a minimal schema marker for downstream steps.
        # Train will still require labels for supervised learning.
        write_json(out_root / "labels_meta.json", {"present": False})
