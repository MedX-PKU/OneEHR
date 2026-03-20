"""Run manifest read/write (merged from run_manifest.py + read.py)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oneehr.config.schema import ExperimentConfig
from oneehr.utils import as_jsonable, ensure_dir, write_json


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    data: dict[str, Any]

    def dynamic_feature_columns(self) -> list[str]:
        cols = (((self.data.get("features") or {}).get("dynamic") or {}).get("feature_columns")) or []
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("Invalid run_manifest: features.dynamic.feature_columns must be list[str]")
        return list(cols)

    def static_feature_columns(self) -> list[str]:
        cols = (((self.data.get("features") or {}).get("static") or {}).get("feature_columns")) or []
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("Invalid run_manifest: features.static.feature_columns must be list[str]")
        return list(cols)

    def static_matrix_path(self) -> Path | None:
        p = (((self.data.get("features") or {}).get("static") or {}).get("matrix_parquet_path")) or None
        if p is None:
            return None
        if not isinstance(p, str) or not p:
            raise ValueError("Invalid run_manifest: features.static.matrix_parquet_path must be str")
        return Path(p)


def read_run_manifest(run_root: Path) -> RunManifest | None:
    path = run_root / "run_manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    v = int(data.get("schema_version", 0) or 0)
    return RunManifest(schema_version=v, data=data)


def write_run_manifest(
    *,
    out_root: Path,
    cfg: ExperimentConfig,
    dynamic_feature_columns: list[str] | None,
    static_feature_columns: list[str] | None,
    static_postprocess_pipeline: list[dict[str, object]] | None,
    patient_tabular_path: str | None,
    time_tabular_path: str | None,
) -> None:
    """Write a run-level manifest describing data + features for reproducibility."""

    out_root = ensure_dir(out_root)

    dyn_cols = [] if not dynamic_feature_columns else list(dynamic_feature_columns)
    st_cols = [] if not static_feature_columns else list(static_feature_columns)

    manifest = {
        "schema_version": 6,
        "dataset": as_jsonable(asdict(cfg.dataset)),
        "task": as_jsonable(asdict(cfg.task)),
        "split": as_jsonable(asdict(cfg.split)),
        "preprocess": {
            "bin_size": str(cfg.preprocess.bin_size),
            "numeric_strategy": str(cfg.preprocess.numeric_strategy),
            "categorical_strategy": str(cfg.preprocess.categorical_strategy),
            "code_selection": str(cfg.preprocess.code_selection),
            "top_k_codes": None if cfg.preprocess.top_k_codes is None else int(cfg.preprocess.top_k_codes),
            "min_code_count": int(cfg.preprocess.min_code_count),
            "pipeline": as_jsonable(list(cfg.preprocess.pipeline)),
        },
        "static": {
            "postprocess": None
            if static_postprocess_pipeline is None
            else {"schema_version": 1, "pipeline": as_jsonable(list(static_postprocess_pipeline))},
        },
        "eval": {
            "instance_unit": str(cfg.eval.instance_unit),
            "max_instances": None if cfg.eval.max_instances is None else int(cfg.eval.max_instances),
            "seed": int(cfg.eval.seed),
            "include_static": bool(cfg.eval.include_static),
            "include_analysis_context": bool(cfg.eval.include_analysis_context),
            "max_events": int(cfg.eval.max_events),
            "time_order": str(cfg.eval.time_order),
            "slice_by": list(cfg.eval.slice_by),
            "primary_metric": None if cfg.eval.primary_metric is None else str(cfg.eval.primary_metric),
            "bootstrap_samples": int(cfg.eval.bootstrap_samples),
            "save_evidence": bool(cfg.eval.save_evidence),
            "save_traces": bool(cfg.eval.save_traces),
            "text_render_template": str(cfg.eval.text_render_template),
            "backends": [
                {
                    "name": str(m.name),
                    "provider": str(m.provider),
                    "base_url": str(m.base_url),
                    "model": str(m.model),
                    "api_key_env": str(m.api_key_env),
                    "system_prompt": None if m.system_prompt is None else str(m.system_prompt),
                    "supports_json_schema": bool(m.supports_json_schema),
                    "prompt_token_cost_per_1k": (
                        None if m.prompt_token_cost_per_1k is None else float(m.prompt_token_cost_per_1k)
                    ),
                    "completion_token_cost_per_1k": (
                        None if m.completion_token_cost_per_1k is None else float(m.completion_token_cost_per_1k)
                    ),
                    "headers": dict(m.headers),
                }
                for m in cfg.eval.backends
            ],
            "systems": [
                {
                    "name": str(system.name),
                    "kind": str(system.kind),
                    "framework_type": None if system.framework_type is None else str(system.framework_type),
                    "enabled": bool(system.enabled),
                    "sample_unit": str(system.sample_unit),
                    "source_model": None if system.source_model is None else str(system.source_model),
                    "backend_refs": list(system.backend_refs),
                    "prompt_template": str(system.prompt_template),
                    "max_samples": None if system.max_samples is None else int(system.max_samples),
                    "max_rounds": int(system.max_rounds),
                    "concurrency": int(system.concurrency),
                    "max_retries": int(system.max_retries),
                    "timeout_seconds": float(system.timeout_seconds),
                    "temperature": float(system.temperature),
                    "top_p": float(system.top_p),
                    "seed": None if system.seed is None else int(system.seed),
                    "framework_params": as_jsonable(dict(system.framework_params)),
                }
                for system in cfg.eval.systems
            ],
            "suites": [
                {
                    "name": str(suite.name),
                    "splits": list(suite.splits),
                    "include_systems": list(suite.include_systems),
                    "primary_metric": None if suite.primary_metric is None else str(suite.primary_metric),
                    "secondary_metrics": list(suite.secondary_metrics),
                    "slice_by": list(suite.slice_by),
                    "min_coverage": float(suite.min_coverage),
                    "compare_pairs": [list(pair) for pair in suite.compare_pairs],
                }
                for suite in cfg.eval.suites
            ],
        },
        "features": {
            "dynamic": {
                "feature_columns": dyn_cols,
            },
            "static": {
                "feature_columns": st_cols,
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
