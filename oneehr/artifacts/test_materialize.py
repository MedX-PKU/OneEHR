from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneehr.artifacts.read import RunManifest
from oneehr.data.binning import bin_events
from oneehr.data.postprocess import FittedPostprocess, transform_postprocess_pipeline
from oneehr.data.static_postprocess import _encode_static_categoricals


@dataclass(frozen=True)
class MaterializedTestViews:
    binned: pd.DataFrame
    labels_df: pd.DataFrame | None
    X: pd.DataFrame
    y: pd.Series | None
    key_df: pd.DataFrame | None


def _load_static_postprocess_from_manifest(manifest: RunManifest) -> FittedPostprocess | None:
    data = manifest.data.get("static") or {}
    post = data.get("postprocess")
    if post is None:
        return None
    pipeline = (post or {}).get("pipeline")
    if not isinstance(pipeline, list):
        raise ValueError("Invalid run_manifest.json: static.postprocess.pipeline must be a list")
    # Backward compat: older manifests stored mean/std/fill as stringified pandas Series.
    fixed: list[dict[str, object]] = []
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        step2 = dict(step)
        for k in ("mean", "std", "fill"):
            v = step2.get(k)
            if isinstance(v, str) and "dtype:" in v and "\n" in v:
                # Parse lines like: "col  value"
                out = {}
                for ln in v.splitlines():
                    ln = ln.strip()
                    if not ln or ln.startswith("dtype:"):
                        continue
                    parts = ln.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        try:
                            out[key] = float(parts[-1])
                        except Exception:
                            continue
                step2[k] = pd.Series(out, dtype=float)
        fixed.append(step2)
    return FittedPostprocess(pipeline=fixed)


def _transform_static_like_train(
    *,
    static_raw: pd.DataFrame | None,
    manifest: RunManifest,
) -> pd.DataFrame | None:
    if static_raw is None or static_raw.empty:
        return None

    if "patient_id" not in static_raw.columns:
        raise ValueError("static.csv missing required column: patient_id")

    raw = static_raw.set_index(static_raw["patient_id"].astype(str), drop=False)
    raw = raw.drop(columns=["patient_id"])

    X0 = _encode_static_categoricals(raw)
    feat_cols = manifest.static_feature_columns()
    X0 = X0.reindex(columns=feat_cols).fillna(0.0)

    fitted = _load_static_postprocess_from_manifest(manifest)
    if fitted is not None:
        X0 = transform_postprocess_pipeline(X0, fitted)

    # Ensure stable column order.
    X0 = X0.reindex(columns=feat_cols)
    X0.index.name = "patient_id"
    return X0


def _build_patient_tabular_from_binned(
    *,
    binned: pd.DataFrame,
    feat_cols: list[str],
) -> pd.DataFrame:
    if binned.empty:
        return pd.DataFrame(columns=["patient_id", *feat_cols]).set_index("patient_id")
    last = (
        binned.sort_values(["patient_id", "bin_time"], kind="stable")
        .groupby("patient_id", sort=False)[feat_cols]
        .last()
    )
    last.index = last.index.astype(str)
    last.index.name = "patient_id"
    return last


def _build_time_tabular_from_binned(
    *,
    binned: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if binned.empty:
        key = pd.DataFrame(columns=["patient_id", "bin_time"])
        X = pd.DataFrame(columns=feat_cols)
        return X, key

    df = binned[["patient_id", "bin_time", *feat_cols]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    key = df[["patient_id", "bin_time"]].reset_index(drop=True)
    X = df[feat_cols].reset_index(drop=True)
    return X, key


def materialize_test_views(
    *,
    manifest: RunManifest,
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    labels_fn: str | None,
) -> MaterializedTestViews:
    """Materialize test-time binned + tabular views using train-time schemas.

    This supports 2 test scenarios:
      1) Offline evaluation (labels available via `label` or `labels_fn`)
      2) Online inference (no labels): returns `y=None`
    """

    mode = str((manifest.data.get("task") or {}).get("prediction_mode", "patient"))
    feat_cols = manifest.dynamic_feature_columns()
    preprocess_cfg = manifest.data.get("preprocess") or {}
    # Build a minimal PreprocessConfig-like dict by reusing ExperimentConfig's schema via optional imports.
    # We avoid depending on ExperimentConfig here to keep test-time logic run-manifest driven.
    from oneehr.config.schema import PreprocessConfig, DynamicTableConfig

    preprocess = PreprocessConfig(
        bin_size=str(preprocess_cfg.get("bin_size", "1d")),
        numeric_strategy=str(preprocess_cfg.get("numeric_strategy", "mean")),
        categorical_strategy=str(preprocess_cfg.get("categorical_strategy", "onehot")),
        code_selection="list",
        top_k_codes=None,
        min_code_count=1,
        code_list=[],
        pipeline=list(preprocess_cfg.get("pipeline") or []),
    )
    # Use dynamic feature columns as a stable schema for test-time alignment.
    # Derive code_list from feature columns (supports num__/cat__).
    code_set: set[str] = set()
    for c in feat_cols:
        if c.startswith("num__"):
            code_set.add(c[len("num__") :])
        elif c.startswith("cat__"):
            # cat__{code}__{level}
            rest = c[len("cat__") :]
            code = rest.split("__", 1)[0]
            if code:
                code_set.add(code)
    preprocess = PreprocessConfig(
        bin_size=preprocess.bin_size,
        numeric_strategy=preprocess.numeric_strategy,
        categorical_strategy=preprocess.categorical_strategy,
        code_selection="list",
        top_k_codes=None,
        min_code_count=1,
        code_list=sorted(code_set),
        pipeline=preprocess.pipeline,
    )

    binned_res = bin_events(dynamic, DynamicTableConfig(path=None), preprocess)
    binned = binned_res.table

    # Align dynamic columns exactly to training feature columns.
    for c in feat_cols:
        if c not in binned.columns:
            binned[c] = 0.0
    binned = binned[["patient_id", "bin_time", "label", *feat_cols]].copy()

    # Labels: if labels_fn provided, compute from (dynamic/static/label) even at test time.
    labels_df = None
    if labels_fn is not None:
        from oneehr.config.schema import (
            DatasetConfig,
            DynamicTableConfig,
            ExperimentConfig,
            LabelTableConfig,
            LabelsConfig,
            ModelConfig,
            OutputConfig,
            PreprocessConfig,
            SplitConfig,
            StaticTableConfig,
            TaskConfig,
            TrainerConfig,
        )
        from oneehr.data.labels import normalize_patient_labels, normalize_time_labels, run_label_fn

        ds = manifest.data.get("dataset") or {}
        task = manifest.data.get("task") or {}
        split = manifest.data.get("split") or {}
        cfg = ExperimentConfig(
            dataset=DatasetConfig(
                dynamic=DynamicTableConfig(path=None),
                static=StaticTableConfig(path=None) if static is not None else None,
                label=LabelTableConfig(path=None) if label is not None else None,
            ),
            task=TaskConfig(kind=str(task.get("kind", "binary")), prediction_mode=str(task.get("prediction_mode", "patient"))),
            split=SplitConfig(kind=str(split.get("kind", "random"))),
            model=ModelConfig(name="xgboost"),
            preprocess=preprocess,
            labels=LabelsConfig(fn=str(labels_fn)),
            trainer=TrainerConfig(),
            output=OutputConfig(),
        )
        labels_res = run_label_fn(dynamic, static, label, cfg)
        if labels_res is not None:
            if mode == "patient":
                labels_df = normalize_patient_labels(labels_res.df)
            else:
                labels_df = normalize_time_labels(labels_res.df, cfg)

    # If a user provided `label.csv` in the training-dataset schema
    # (patient_id,label_time,label_code,label_value), treat it as raw labels input
    # and rely on `labels_fn` to normalize it. Only accept already-normalized label
    # tables when `labels_fn` is NOT provided.
    if labels_fn is None and label is not None and not label.empty:
        # Normalize shape based on manifest prediction_mode.
        if mode == "patient":
            if "patient_id" not in label.columns or "label" not in label.columns:
                raise ValueError("label.csv must contain columns: patient_id, label")
            labels_df = label[["patient_id", "label"]].copy()
            labels_df["patient_id"] = labels_df["patient_id"].astype(str)
        else:
            if not {"patient_id", "bin_time", "label"}.issubset(set(label.columns)):
                raise ValueError("label.csv must contain columns: patient_id, bin_time, label (for time mode)")
            cols = ["patient_id", "bin_time", "label"]
            if "mask" in label.columns:
                cols.append("mask")
            labels_df = label[cols].copy()
            labels_df["patient_id"] = labels_df["patient_id"].astype(str)

    # Static
    static_all = _transform_static_like_train(static_raw=static, manifest=manifest)

    if mode == "patient":
        X_dyn = _build_patient_tabular_from_binned(binned=binned, feat_cols=feat_cols)
        if static_all is not None:
            # Join by patient_id.
            # Avoid collisions when a feature exists in both dynamic and static spaces.
            overlap = [c for c in static_all.columns if c in X_dyn.columns]
            if overlap:
                static_all = static_all.drop(columns=overlap)
            X = X_dyn.join(static_all, how="left").fillna(0.0)
        else:
            X = X_dyn
        if labels_df is None:
            return MaterializedTestViews(binned=binned, labels_df=None, X=X, y=None, key_df=None)
        y = labels_df.set_index("patient_id")["label"].reindex(X.index).astype(float)
        return MaterializedTestViews(binned=binned, labels_df=labels_df, X=X, y=y, key_df=None)

    # time mode
    X_dyn, key = _build_time_tabular_from_binned(binned=binned, feat_cols=feat_cols)
    if static_all is not None and not key.empty:
        static_join = static_all.reindex(key["patient_id"].astype(str)).reset_index(drop=True)
        X = pd.concat([X_dyn.reset_index(drop=True), static_join.reset_index(drop=True)], axis=1)
    else:
        X = X_dyn
    y = None
    if labels_df is not None and not key.empty:
        merged = key.merge(labels_df[["patient_id", "bin_time", "label"]], on=["patient_id", "bin_time"], how="left")
        y = merged["label"].astype(float)
    return MaterializedTestViews(binned=binned, labels_df=labels_df, X=X, y=y, key_df=key)
