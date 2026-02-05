from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from oneehr.config.schema import ExperimentConfig
from oneehr.utils.io import ensure_dir, write_json


def _sha256_lines(lines: list[str]) -> str:
    import hashlib

    norm = "\n".join([ln.strip() for ln in lines]) + "\n"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _as_jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_as_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _as_jsonable(x) for k, x in v.items()}
    return str(v)


def write_run_manifest(
    *,
    out_root: Path,
    cfg: ExperimentConfig,
    dynamic_feature_columns: list[str] | None,
    static_raw_cols: list[str] | None,
    static_feature_columns: list[str] | None,
    static_postprocess_pipeline: list[dict[str, object]] | None,
    patient_tabular_path: str | None,
    time_tabular_path: str | None,
) -> None:
    """Write a run-level manifest describing data + features for reproducibility.

    This is a v2 schema that aims to unify tabular/DL pipelines and static/dynamic features.
    It is intended as the single source of truth for a training run.
    """

    out_root = ensure_dir(out_root)

    dyn_cols = [] if not dynamic_feature_columns else list(dynamic_feature_columns)
    st_cols = [] if not static_feature_columns else list(static_feature_columns)

    manifest = {
        "schema_version": 2,
        "dataset": _as_jsonable(asdict(cfg.dataset)),
        "task": _as_jsonable(asdict(cfg.task)),
        "split": _as_jsonable(asdict(cfg.split)),
        "preprocess": {
            "bin_size": str(cfg.preprocess.bin_size),
            "numeric_strategy": str(cfg.preprocess.numeric_strategy),
            "categorical_strategy": str(cfg.preprocess.categorical_strategy),
            "code_selection": str(cfg.preprocess.code_selection),
            "top_k_codes": None if cfg.preprocess.top_k_codes is None else int(cfg.preprocess.top_k_codes),
            "min_code_count": int(cfg.preprocess.min_code_count),
            "pipeline": _as_jsonable(list(cfg.preprocess.pipeline)),
        },
        "features": {
            "dynamic": {
                "feature_columns": dyn_cols,
                "feature_columns_json_path": "features/dynamic/feature_columns.json",
            },
            "static": {
                "feature_columns": st_cols,
                "feature_columns_json_path": "features/static/feature_columns.json",
                "matrix_parquet_path": None if not st_cols else "features/static/static_all.parquet",
            },
        },
        "artifacts": {
            "binned_parquet_path": "binned.parquet",
            "labels_parquet_path": "labels.parquet",
            "patient_tabular_parquet_path": patient_tabular_path,
            "time_tabular_parquet_path": time_tabular_path,
        },
    }

    write_json(out_root / "run_manifest.json", manifest)
