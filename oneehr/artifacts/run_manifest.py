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

    This is a v4 schema that captures the run contract for modeling, analysis,
    cases, and agent workflows in one manifest.
    It is intended as the single source of truth for a training run.
    """

    out_root = ensure_dir(out_root)

    dyn_cols = [] if not dynamic_feature_columns else list(dynamic_feature_columns)
    st_cols = [] if not static_feature_columns else list(static_feature_columns)

    manifest = {
        "schema_version": 4,
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
        "cases": {
            "include_static": bool(cfg.cases.include_static),
            "include_analysis_refs": bool(cfg.cases.include_analysis_refs),
            "history_window": None if cfg.cases.history_window is None else str(cfg.cases.history_window),
            "max_events": int(cfg.cases.max_events),
            "time_order": str(cfg.cases.time_order),
            "case_limit": None if cfg.cases.case_limit is None else int(cfg.cases.case_limit),
        },
        "agent": {
            "predict": {
                "enabled": bool(cfg.agent.predict.enabled),
                "sample_unit": str(cfg.agent.predict.sample_unit),
                "prompt_template": str(cfg.agent.predict.prompt_template),
                "json_schema_version": int(cfg.agent.predict.json_schema_version),
                "max_samples": None if cfg.agent.predict.max_samples is None else int(cfg.agent.predict.max_samples),
                "save_prompts": bool(cfg.agent.predict.save_prompts),
                "save_responses": bool(cfg.agent.predict.save_responses),
                "save_parsed": bool(cfg.agent.predict.save_parsed),
                "concurrency": int(cfg.agent.predict.concurrency),
                "max_retries": int(cfg.agent.predict.max_retries),
                "timeout_seconds": float(cfg.agent.predict.timeout_seconds),
                "temperature": float(cfg.agent.predict.temperature),
                "top_p": float(cfg.agent.predict.top_p),
                "seed": None if cfg.agent.predict.seed is None else int(cfg.agent.predict.seed),
                "prompt": {
                    "include_static": bool(cfg.agent.predict.prompt.include_static),
                    "include_labels_context": bool(cfg.agent.predict.prompt.include_labels_context),
                    "history_window": None if cfg.agent.predict.prompt.history_window is None else str(cfg.agent.predict.prompt.history_window),
                    "max_events": int(cfg.agent.predict.prompt.max_events),
                    "time_order": str(cfg.agent.predict.prompt.time_order),
                    "sections": list(cfg.agent.predict.prompt.sections),
                },
                "output": {
                    "include_explanation": bool(cfg.agent.predict.output.include_explanation),
                    "include_confidence": bool(cfg.agent.predict.output.include_confidence),
                },
                "backends": [
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
                    for m in cfg.agent.predict.backends
                ],
            },
            "review": {
                "enabled": bool(cfg.agent.review.enabled),
                "prompt_template": str(cfg.agent.review.prompt_template),
                "json_schema_version": int(cfg.agent.review.json_schema_version),
                "prediction_origins": list(cfg.agent.review.prediction_origins),
                "max_cases": None if cfg.agent.review.max_cases is None else int(cfg.agent.review.max_cases),
                "save_prompts": bool(cfg.agent.review.save_prompts),
                "save_responses": bool(cfg.agent.review.save_responses),
                "save_parsed": bool(cfg.agent.review.save_parsed),
                "concurrency": int(cfg.agent.review.concurrency),
                "max_retries": int(cfg.agent.review.max_retries),
                "timeout_seconds": float(cfg.agent.review.timeout_seconds),
                "temperature": float(cfg.agent.review.temperature),
                "top_p": float(cfg.agent.review.top_p),
                "seed": None if cfg.agent.review.seed is None else int(cfg.agent.review.seed),
                "prompt": {
                    "include_static": bool(cfg.agent.review.prompt.include_static),
                    "include_ground_truth": bool(cfg.agent.review.prompt.include_ground_truth),
                    "include_analysis_context": bool(cfg.agent.review.prompt.include_analysis_context),
                    "max_events": int(cfg.agent.review.prompt.max_events),
                    "time_order": str(cfg.agent.review.prompt.time_order),
                    "sections": list(cfg.agent.review.prompt.sections),
                },
                "backends": [
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
                    for m in cfg.agent.review.backends
                ],
            },
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
