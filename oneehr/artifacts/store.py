"""RunIO, inference views, and label validation (merged from run_io.py + inference.py + labels.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.artifacts.manifest import RunManifest, read_run_manifest
from oneehr.data.binning import bin_events
from oneehr.data.tabular import (
    FittedPostprocess,
    _encode_static_categoricals,
    transform_postprocess_pipeline,
)


# ─── Label validation ────────────────────────────────────────────────────────


def validate_patient_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "label"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"labels.parquet missing columns: {missing}")
    out = df[["patient_id", "label"]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    return out


def validate_time_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "bin_time", "label"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"labels.parquet missing columns: {missing}")
    cols = ["patient_id", "bin_time", "label"]
    if "mask" in df.columns:
        cols.append("mask")
    out = df[cols].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    return out


# ─── RunIO ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunIO:
    run_root: Path

    @classmethod
    def from_cfg(cls, *, root: Path, run_name: str) -> "RunIO":
        return cls(run_root=Path(root) / run_name)

    def require_manifest(self):
        manifest = read_run_manifest(self.run_root)
        if manifest is None:
            raise SystemExit(
                f"Missing run_manifest.json under {self.run_root}. "
                "Run `oneehr preprocess` first."
            )
        return manifest

    def load_binned(self, manifest) -> pd.DataFrame:
        p = (manifest.data.get("artifacts") or {}).get("binned_parquet_path")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing binned_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        required = {"patient_id", "bin_time"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Invalid binned.parquet: missing columns {missing}")

        feat_cols = manifest.dynamic_feature_columns()
        missing_feat = [c for c in feat_cols if c not in df.columns]
        if missing_feat:
            raise SystemExit(f"Invalid binned.parquet: missing feature columns {missing_feat}")

        base = ["patient_id", "bin_time"]
        if "label" in df.columns:
            base.append("label")
        df = df[base + feat_cols]
        return df

    def load_patient_view(self, manifest) -> tuple[pd.DataFrame, pd.Series]:
        p = (manifest.data.get("artifacts") or {}).get("patient_tabular_parquet_path")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing patient_tabular_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        if "patient_id" not in df.columns or "label" not in df.columns:
            raise SystemExit("Invalid patient_tabular.parquet: missing patient_id/label.")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        feat_cols = manifest.dynamic_feature_columns()
        missing = [c for c in ["patient_id", "label", *feat_cols] if c not in df.columns]
        if missing:
            raise SystemExit(f"Invalid patient_tabular.parquet: missing columns {missing}")
        df = df[["patient_id", "label", *feat_cols]]
        X = df[["patient_id", *feat_cols]].set_index("patient_id")
        y = df["label"]
        return X, y

    def load_time_view(self, manifest) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        p = (manifest.data.get("artifacts") or {}).get("time_tabular_parquet_path")
        if not isinstance(p, str) or not p:
            raise SystemExit("Missing time_tabular_parquet_path in run_manifest.json. Re-run `oneehr preprocess`.")
        df = pd.read_parquet(self.run_root / p)
        required = {"patient_id", "bin_time", "label"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Invalid time_tabular.parquet: missing columns {missing}")
        feat_cols = manifest.dynamic_feature_columns()
        missing2 = [c for c in feat_cols if c not in df.columns]
        if missing2:
            raise SystemExit(f"Invalid time_tabular.parquet: missing feature columns {missing2}")
        df = df[["patient_id", "bin_time", "label", *feat_cols]].reset_index(drop=True)
        key = df[["patient_id", "bin_time"]]
        y = df["label"]
        X = df[feat_cols]
        return X, y, key

    def load_static_all(self, manifest) -> tuple[pd.DataFrame | None, list[str] | None]:
        st_path = manifest.static_matrix_path()
        if st_path is None:
            return None, None
        static_all = pd.read_parquet(self.run_root / st_path)
        cols = manifest.static_feature_columns()
        if list(static_all.columns) != list(cols):
            raise SystemExit(
                "Static feature_columns mismatch with run_manifest.json. "
                "Re-run `oneehr preprocess`."
            )
        return static_all, cols

    def load_labels(self, manifest) -> pd.DataFrame | None:
        p = (manifest.data.get("artifacts") or {}).get("labels_parquet_path")
        if p is None:
            return None
        if not isinstance(p, str) or not p:
            raise SystemExit("Invalid labels_parquet_path in run_manifest.json.")
        path = self.run_root / p
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        mode = str((manifest.data.get("task") or {}).get("prediction_mode", "patient"))
        if mode == "patient":
            return validate_patient_labels(df)
        if mode == "time":
            return validate_time_labels(df)
        raise ValueError(f"Unsupported prediction_mode in run_manifest.json: {mode!r}")

    def load_fitted_postprocess(self, split_name: str):
        pp_path = self.run_root / "preprocess" / split_name / "pipeline.json"
        if not pp_path.exists():
            return None
        data = json.loads(pp_path.read_text(encoding="utf-8"))
        pipeline = data.get("pipeline")
        if not isinstance(pipeline, list):
            return None
        return FittedPostprocess(pipeline=pipeline)


# ─── Test-time inference views ────────────────────────────────────────────────


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
    if not all(isinstance(step, dict) for step in pipeline):
        raise ValueError("Invalid run_manifest.json: static.postprocess.pipeline must be a list of dicts")
    return FittedPostprocess(pipeline=list(pipeline))


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
    id_like_cols = [c for c in raw.columns if str(c).lower() in {"patient_id", "patientid"}]
    raw = raw.drop(columns=id_like_cols, errors="ignore")

    X0 = _encode_static_categoricals(raw)
    feat_cols = manifest.static_feature_columns()
    X0 = X0.reindex(columns=feat_cols).fillna(0.0)

    fitted = _load_static_postprocess_from_manifest(manifest)
    if fitted is not None:
        X0 = transform_postprocess_pipeline(X0, fitted)

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
    """Materialize test-time binned + tabular views using train-time schemas."""

    mode = str((manifest.data.get("task") or {}).get("prediction_mode", "patient"))
    feat_cols = manifest.dynamic_feature_columns()
    preprocess_cfg = manifest.data.get("preprocess") or {}
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
    code_set: set[str] = set()
    for c in feat_cols:
        if c.startswith("num__"):
            code_set.add(c[len("num__"):])
        elif c.startswith("cat__"):
            rest = c[len("cat__"):]
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

    for c in feat_cols:
        if c not in binned.columns:
            binned[c] = 0.0
    binned = binned[["patient_id", "bin_time", "label", *feat_cols]].copy()

    labels_df = None
    if labels_fn is not None:
        from oneehr.config.schema import (
            DatasetConfig,
            ExperimentConfig,
            LabelTableConfig,
            LabelsConfig,
            ModelConfig,
            OutputConfig,
            SplitConfig,
            StaticTableConfig,
            TaskConfig,
            TrainerConfig,
        )
        from oneehr.data.labels import normalize_patient_labels, normalize_time_labels, run_label_fn

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

    if labels_fn is None and label is not None and not label.empty:
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

    static_all = _transform_static_like_train(static_raw=static, manifest=manifest)

    if mode == "patient":
        X_dyn = _build_patient_tabular_from_binned(binned=binned, feat_cols=feat_cols)
        if static_all is not None:
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
