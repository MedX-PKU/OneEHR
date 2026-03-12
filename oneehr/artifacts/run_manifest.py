from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from oneehr.config.schema import ExperimentConfig
from oneehr.utils.io import as_jsonable, ensure_dir, write_json


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

    This is a v3 schema that aims to unify tabular/DL pipelines and static/dynamic features,
    while capturing LLM workflow settings in the same run contract.
    It is intended as the single source of truth for a training run.
    """

    out_root = ensure_dir(out_root)

    dyn_cols = [] if not dynamic_feature_columns else list(dynamic_feature_columns)
    st_cols = [] if not static_feature_columns else list(static_feature_columns)

    manifest = {
        "schema_version": 3,
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
        "llm": {
            "enabled": bool(cfg.llm.enabled),
            "sample_unit": str(cfg.llm.sample_unit),
            "prompt_template": str(cfg.llm.prompt_template),
            "json_schema_version": int(cfg.llm.json_schema_version),
            "max_samples": None if cfg.llm.max_samples is None else int(cfg.llm.max_samples),
            "save_prompts": bool(cfg.llm.save_prompts),
            "save_responses": bool(cfg.llm.save_responses),
            "save_parsed": bool(cfg.llm.save_parsed),
            "concurrency": int(cfg.llm.concurrency),
            "max_retries": int(cfg.llm.max_retries),
            "timeout_seconds": float(cfg.llm.timeout_seconds),
            "temperature": float(cfg.llm.temperature),
            "top_p": float(cfg.llm.top_p),
            "seed": None if cfg.llm.seed is None else int(cfg.llm.seed),
            "prompt": {
                "include_static": bool(cfg.llm.prompt.include_static),
                "include_labels_context": bool(cfg.llm.prompt.include_labels_context),
                "history_window": None if cfg.llm.prompt.history_window is None else str(cfg.llm.prompt.history_window),
                "max_events": int(cfg.llm.prompt.max_events),
                "time_order": str(cfg.llm.prompt.time_order),
                "sections": list(cfg.llm.prompt.sections),
            },
            "output": {
                "include_explanation": bool(cfg.llm.output.include_explanation),
                "include_confidence": bool(cfg.llm.output.include_confidence),
            },
            "models": [
                {
                    "name": str(m.name),
                    "provider": str(m.provider),
                    "base_url": str(m.base_url),
                    "model": str(m.model),
                    "api_key_env": str(m.api_key_env),
                    "system_prompt": None if m.system_prompt is None else str(m.system_prompt),
                    "supports_json_schema": bool(m.supports_json_schema),
                    "headers": dict(m.headers),
                }
                for m in cfg.llm_models
            ],
        },
        "workspace": {
            "include_static": bool(cfg.workspace.include_static),
            "include_analysis_refs": bool(cfg.workspace.include_analysis_refs),
            "history_window": None if cfg.workspace.history_window is None else str(cfg.workspace.history_window),
            "max_events": int(cfg.workspace.max_events),
            "time_order": str(cfg.workspace.time_order),
            "case_limit": None if cfg.workspace.case_limit is None else int(cfg.workspace.case_limit),
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
