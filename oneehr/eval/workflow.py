from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from oneehr.agent.client import OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec, ParsedPrediction
from oneehr.agent.schema import build_response_format, parse_prediction_response, schema_prompt_text
from oneehr.agent.runtime import execute_agent_request, run_jobs
from oneehr.agent.templates import safe_case_slug, select_events
from oneehr.analysis.reporting import describe_patient_case, read_analysis_index
from oneehr.artifacts.store import RunIO
from oneehr.config.schema import EvalBackendConfig, EvalSystemConfig, ExperimentConfig
from oneehr.data.io import load_dynamic_table_optional, load_static_table
from oneehr.data.splits import require_saved_splits
from oneehr.data.test_samples import build_test_sample_frame
from oneehr.eval.bootstrap import bootstrap_metric
from oneehr.eval.metrics import binary_metrics, regression_metrics
from oneehr.utils import as_jsonable, ensure_dir, write_json

EVAL_SCHEMA_VERSION = 1
FRAMEWORK_TYPES = {
    "single_llm",
    "healthcareagent",
    "reconcile",
    "mac",
    "medagent",
    "colacare",
    "mdagents",
}
DEFAULT_SPECIALTIES = [
    "general_medicine",
    "cardiology",
    "radiology",
    "critical_care",
    "infectious_disease",
]


@dataclass(frozen=True)
class BuiltEvalArtifacts:
    index_path: Path
    instance_count: int


@dataclass(frozen=True)
class RanEvalSystems:
    summary_path: Path
    system_count: int


@dataclass(frozen=True)
class BuiltEvalReport:
    summary_path: Path
    leaderboard_rows: int


@dataclass(frozen=True)
class ParsedStageJson:
    parsed_ok: bool
    payload: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class StageExecution:
    parsed: ParsedStageJson | ParsedPrediction | None
    trace_row: dict[str, Any]


def build_eval_artifacts(cfg: ExperimentConfig, *, run_root: Path, force: bool = False) -> BuiltEvalArtifacts:
    eval_root = ensure_dir(run_root / "eval")
    if force:
        for rel in ["instances", "evidence", "index.json"]:
            path = eval_root / rel
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()

    frame = _build_eval_instance_frame(cfg, run_root=run_root)
    instance_dir = ensure_dir(eval_root / "instances")
    instance_path = instance_dir / "instances.parquet"
    frame.to_parquet(instance_path, index=False)

    dynamic_by_patient = _load_dynamic_by_patient(cfg)
    static_by_patient = _load_static_by_patient(cfg) if cfg.eval.include_static else {}
    evidence_rows: list[dict[str, Any]] = []
    if cfg.eval.save_evidence:
        evidence_rows = _materialize_evidence_bundles(
            cfg,
            run_root=run_root,
            frame=frame,
            dynamic_by_patient=dynamic_by_patient,
            static_by_patient=static_by_patient,
        )
    else:
        evidence_rows = [
            {
                "instance_id": str(row["instance_id"]),
                "patient_id": str(row["patient_id"]),
                "split": str(row["split"]),
                "split_role": str(row.get("split_role", "test")),
                "prediction_mode": str(row["prediction_mode"]),
                "bin_time": _json_time(row.get("bin_time")),
                "ground_truth": row.get("ground_truth"),
                "evidence_path": None,
            }
            for row in frame.to_dict(orient="records")
        ]

    index_path = eval_root / "index.json"
    write_json(
        index_path,
        {
            "schema_version": EVAL_SCHEMA_VERSION,
            "run_dir": str(run_root),
            "task": {"kind": str(cfg.task.kind), "prediction_mode": str(cfg.task.prediction_mode)},
            "eval": {
                "instance_unit": str(cfg.eval.instance_unit),
                "max_instances": None if cfg.eval.max_instances is None else int(cfg.eval.max_instances),
                "save_evidence": bool(cfg.eval.save_evidence),
                "max_events": int(cfg.eval.max_events),
                "time_order": str(cfg.eval.time_order),
                "include_static": bool(cfg.eval.include_static),
                "include_analysis_context": bool(cfg.eval.include_analysis_context),
                "text_render_template": str(cfg.eval.text_render_template),
                "instance_path": str(instance_path.relative_to(run_root)),
            },
            "instance_count": int(len(frame)),
            "records": evidence_rows,
        },
    )
    return BuiltEvalArtifacts(index_path=index_path, instance_count=len(frame))


def run_eval_systems(cfg: ExperimentConfig, *, run_root: Path, force: bool = False) -> RanEvalSystems:
    index = _ensure_eval_index(cfg, run_root=run_root)
    eval_root = ensure_dir(run_root / "eval")
    summary_path = eval_root / "summary.json"
    if force and summary_path.exists():
        summary_path.unlink()
    systems_root = ensure_dir(eval_root / "predictions")
    traces_root = ensure_dir(eval_root / "traces")

    instance_path = run_root / str(index["eval"]["instance_path"])
    instances = pd.read_parquet(instance_path)
    if instances.empty:
        write_json(summary_path, {"records": []})
        return RanEvalSystems(summary_path=summary_path, system_count=0)

    client = OpenAICompatibleAgentClient()
    backends = {backend.name: backend for backend in cfg.eval.backends}
    system_records: list[dict[str, Any]] = []
    for system in [system for system in cfg.eval.systems if system.enabled]:
        pred_dir = systems_root / system.name
        trace_dir = traces_root / system.name
        if force:
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            if trace_dir.exists():
                shutil.rmtree(trace_dir)
        ensure_dir(pred_dir)
        ensure_dir(trace_dir)

        block = instances.copy()
        if system.max_samples is not None:
            block = block.head(int(system.max_samples)).reset_index(drop=True)

        if system.kind == "trained_model":
            pred_frame = _materialize_trained_model_predictions(cfg, run_root=run_root, system=system, instances=block)
            trace_frame = pd.DataFrame()
        else:
            pred_frame, trace_frame = _run_framework_predictions(
                cfg,
                run_root=run_root,
                system=system,
                instances=block,
                index=index,
                backends=backends,
                client=client,
            )

        pred_path = pred_dir / "predictions.parquet"
        pred_frame.to_parquet(pred_path, index=False)
        trace_path = None
        if cfg.eval.save_traces and not trace_frame.empty:
            trace_path = trace_dir / "trace.parquet"
            trace_frame.to_parquet(trace_path, index=False)

        parsed_ok_rows = int((pred_frame["parsed_ok"] == True).sum()) if "parsed_ok" in pred_frame.columns else 0  # noqa: E712
        coverage = float(parsed_ok_rows / len(pred_frame)) if len(pred_frame) else 0.0
        token_total = int(pd.to_numeric(pred_frame.get("token_usage_total"), errors="coerce").fillna(0).sum())
        cost_total = float(pd.to_numeric(pred_frame.get("cost_usd"), errors="coerce").fillna(0.0).sum())
        latency_mean = float(pd.to_numeric(pred_frame.get("latency_ms"), errors="coerce").mean()) if not pred_frame.empty else 0.0
        system_records.append(
            {
                "system_name": system.name,
                "system_kind": system.kind,
                "framework_type": system.framework_type,
                "row_count": int(len(pred_frame)),
                "parsed_ok_rows": parsed_ok_rows,
                "coverage": coverage,
                "mean_latency_ms": latency_mean,
                "total_tokens": token_total,
                "total_cost_usd": cost_total,
                "artifacts": {
                    "predictions_parquet": str(pred_path.relative_to(run_root)),
                    "trace_parquet": None if trace_path is None else str(trace_path.relative_to(run_root)),
                },
            }
        )

    write_json(
        summary_path,
        {
            "schema_version": EVAL_SCHEMA_VERSION,
            "run_dir": str(run_root),
            "task": {"kind": str(cfg.task.kind), "prediction_mode": str(cfg.task.prediction_mode)},
            "records": system_records,
        },
    )
    return RanEvalSystems(summary_path=summary_path, system_count=len(system_records))


def build_eval_report(cfg: ExperimentConfig, *, run_root: Path, force: bool = False) -> BuiltEvalReport:
    summary = _ensure_eval_summary(cfg, run_root=run_root)
    eval_root = ensure_dir(run_root / "eval")
    report_root = ensure_dir(eval_root / "reports")
    leaderboard_path = report_root / "leaderboard.csv"
    split_metrics_path = report_root / "split_metrics.csv"
    pairwise_path = report_root / "pairwise.csv"
    summary_path = report_root / "summary.json"

    if force:
        for path in [leaderboard_path, split_metrics_path, pairwise_path, summary_path]:
            if path.exists():
                path.unlink()

    if not summary.get("records"):
        write_json(summary_path, {"leaderboard_rows": 0, "pairwise_rows": 0})
        return BuiltEvalReport(summary_path=summary_path, leaderboard_rows=0)

    primary_metric = cfg.eval.primary_metric or ("auroc" if cfg.task.kind == "binary" else "rmse")
    prediction_frames: dict[str, pd.DataFrame] = {}
    summary_records = summary.get("records", [])
    for record in summary_records:
        if not isinstance(record, dict):
            continue
        system_name = str(record["system_name"])
        pred_path = run_root / str(((record.get("artifacts") or {}).get("predictions_parquet")))
        prediction_frames[system_name] = pd.read_parquet(pred_path)

    leaderboard_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for record in summary_records:
        if not isinstance(record, dict):
            continue
        system_name = str(record["system_name"])
        frame = prediction_frames.get(system_name, pd.DataFrame())
        system_metrics = _score_prediction_frame(frame, task_kind=cfg.task.kind)
        leaderboard_row = {
            "system_name": system_name,
            "system_kind": str(record.get("system_kind")),
            "framework_type": record.get("framework_type"),
            "row_count": int(len(frame)),
            "coverage": system_metrics["coverage"],
            "scored_rows": system_metrics["scored_rows"],
            "mean_latency_ms": record.get("mean_latency_ms"),
            "total_tokens": record.get("total_tokens"),
            "total_cost_usd": record.get("total_cost_usd"),
            **system_metrics["metrics"],
        }
        leaderboard_rows.append(leaderboard_row)

        if not frame.empty and "split" in frame.columns:
            for split_name, block in frame.groupby("split", sort=True):
                split_metrics = _score_prediction_frame(block, task_kind=cfg.task.kind)
                split_rows.append(
                    {
                        "system_name": system_name,
                        "system_kind": str(record.get("system_kind")),
                        "framework_type": record.get("framework_type"),
                        "split": str(split_name),
                        "row_count": int(len(block)),
                        "coverage": split_metrics["coverage"],
                        "scored_rows": split_metrics["scored_rows"],
                        **split_metrics["metrics"],
                    }
                )

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    if not leaderboard_df.empty and primary_metric in leaderboard_df.columns:
        ascending = cfg.task.kind != "binary" and primary_metric in {"mae", "mse", "rmse", "logloss"}
        leaderboard_df = leaderboard_df.sort_values(primary_metric, ascending=ascending, kind="stable").reset_index(drop=True)
    leaderboard_df.to_csv(leaderboard_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_metrics_path, index=False)

    pairwise_rows = _build_pairwise_rows(
        cfg,
        prediction_frames=prediction_frames,
        primary_metric=primary_metric,
    )
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)

    write_json(
        summary_path,
        {
            "schema_version": EVAL_SCHEMA_VERSION,
            "primary_metric": primary_metric,
            "leaderboard_rows": int(len(leaderboard_df)),
            "pairwise_rows": int(len(pairwise_rows)),
            "artifacts": {
                "leaderboard_csv": str(leaderboard_path.relative_to(run_root)),
                "split_metrics_csv": str(split_metrics_path.relative_to(run_root)),
                "pairwise_csv": str(pairwise_path.relative_to(run_root)),
            },
        },
    )
    return BuiltEvalReport(summary_path=summary_path, leaderboard_rows=len(leaderboard_df))


def _build_eval_instance_frame(cfg: ExperimentConfig, *, run_root: Path) -> pd.DataFrame:
    run = RunIO(run_root=run_root)
    manifest = run.require_manifest()
    labels_df = run.load_labels(manifest)
    splits = require_saved_splits(run_root / "splits", context="running `oneehr eval build`")

    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static = load_static_table(train_dataset.static)
    frame = build_test_sample_frame(
        splits=splits,
        labels_df=labels_df,
        dynamic=dynamic,
        static=static,
        task_kind=cfg.task.kind,
        prediction_mode=cfg.eval.instance_unit,
    )
    if frame.empty:
        return pd.DataFrame(columns=["instance_id", "split", "split_role", "patient_id", "prediction_mode", "task_kind"])
    frame = frame.rename(columns={"sample_id": "instance_id"}).copy()
    if cfg.eval.max_instances is not None:
        frame = frame.head(int(cfg.eval.max_instances)).reset_index(drop=True)
    return frame


def _materialize_evidence_bundles(
    cfg: ExperimentConfig,
    *,
    run_root: Path,
    frame: pd.DataFrame,
    dynamic_by_patient: dict[str, pd.DataFrame],
    static_by_patient: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    eval_root = ensure_dir(run_root / "eval" / "evidence")
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        instance_id = str(row["instance_id"])
        patient_id = str(row["patient_id"])
        anchor_time = _optional_timestamp(row.get("bin_time"))
        events = select_events(
            dynamic=dynamic_by_patient.get(patient_id),
            anchor_time=anchor_time,
            history_window=None,
            max_events=int(cfg.eval.max_events),
            time_order=str(cfg.eval.time_order),
        )
        static_row = static_by_patient.get(patient_id)
        analysis_refs = (
            _collect_analysis_refs(run_root=run_root, patient_id=patient_id)
            if cfg.eval.include_analysis_context
            else {"modules": [], "patient_case_matches": []}
        )
        instance_dir = ensure_dir(eval_root / _instance_dir_name(instance_id))
        events_path = instance_dir / "events.csv"
        static_path = instance_dir / "static.json"
        refs_path = instance_dir / "analysis_refs.json"
        evidence_path = instance_dir / "evidence.json"

        out_events = events.copy() if not events.empty else pd.DataFrame(columns=["event_time", "code", "value"])
        if not out_events.empty:
            out_events["patient_id"] = patient_id
            out_events[["patient_id", "event_time", "code", "value"]].to_csv(events_path, index=False)
        else:
            pd.DataFrame(columns=["patient_id", "event_time", "code", "value"]).to_csv(events_path, index=False)
        write_json(static_path, {"patient_id": patient_id, "features": _series_to_dict(static_row)})
        write_json(refs_path, analysis_refs)
        evidence_payload = {
            "instance_id": instance_id,
            "patient_id": patient_id,
            "split": str(row["split"]),
            "split_role": str(row.get("split_role", "test")),
            "prediction_mode": str(row["prediction_mode"]),
            "task_kind": str(row["task_kind"]),
            "bin_time": None if anchor_time is None else anchor_time.isoformat(),
            "ground_truth": row.get("ground_truth"),
            "event_count": int(len(events)),
            "static_feature_count": int(len(_series_to_dict(static_row))),
            "artifacts": {
                "events_csv": str(events_path.relative_to(run_root)),
                "static_json": str(static_path.relative_to(run_root)),
                "analysis_refs_json": str(refs_path.relative_to(run_root)),
            },
        }
        write_json(evidence_path, as_jsonable(evidence_payload))
        records.append(
            {
                "instance_id": instance_id,
                "patient_id": patient_id,
                "split": str(row["split"]),
                "split_role": str(row.get("split_role", "test")),
                "prediction_mode": str(row["prediction_mode"]),
                "bin_time": None if anchor_time is None else anchor_time.isoformat(),
                "ground_truth": row.get("ground_truth"),
                "event_count": int(len(events)),
                "static_feature_count": int(len(_series_to_dict(static_row))),
                "evidence_path": str(evidence_path.relative_to(run_root)),
            }
        )
    return records


def _materialize_trained_model_predictions(
    cfg: ExperimentConfig,
    *,
    run_root: Path,
    system: EvalSystemConfig,
    instances: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, block in instances.groupby("split", sort=True):
        pred_path = run_root / "preds" / str(system.source_model) / f"{split_name}.parquet"
        if not pred_path.exists():
            for item in block.to_dict(orient="records"):
                rows.append(_empty_prediction_row(cfg, system=system, instance=item, error_code="missing_preds", error_message=f"missing {pred_path}"))
            continue

        preds = pd.read_parquet(pred_path).copy()
        preds["patient_id"] = preds["patient_id"].astype(str)
        if cfg.eval.instance_unit == "time" and "bin_time" in preds.columns:
            preds["bin_time"] = pd.to_datetime(preds["bin_time"], errors="coerce")

        if cfg.eval.instance_unit == "patient":
            merged = block.merge(preds, on="patient_id", how="left", suffixes=("", "_pred"))
        else:
            block = block.copy()
            block["bin_time"] = pd.to_datetime(block["bin_time"], errors="coerce")
            merged = block.merge(preds, on=["patient_id", "bin_time"], how="left", suffixes=("", "_pred"))

        for row in merged.to_dict(orient="records"):
            y_pred = row.get("y_pred")
            parsed_ok = y_pred is not None and pd.notna(y_pred)
            probability = float(y_pred) if parsed_ok and cfg.task.kind == "binary" else None
            value = float(y_pred) if parsed_ok and cfg.task.kind == "regression" else None
            label = int(float(y_pred) >= 0.5) if probability is not None else None
            rows.append(
                {
                    "instance_id": str(row["instance_id"]),
                    "patient_id": str(row["patient_id"]),
                    "split": str(row["split"]),
                    "split_role": str(row.get("split_role", "test")),
                    "bin_time": _json_time(row.get("bin_time")),
                    "ground_truth": row.get("ground_truth"),
                    "system_name": system.name,
                    "system_kind": system.kind,
                    "framework_type": system.framework_type,
                    "parsed_ok": bool(parsed_ok),
                    "prediction": label if cfg.task.kind == "binary" else value,
                    "label": label,
                    "probability": probability,
                    "value": value,
                    "explanation": None,
                    "confidence": None,
                    "latency_ms": None,
                    "token_usage_prompt": 0,
                    "token_usage_completion": 0,
                    "token_usage_total": 0,
                    "cost_usd": 0.0,
                    "round_count": 0,
                    "trace_row_count": 0,
                    "config_sha256": _system_config_sha256(system),
                    "error_code": None if parsed_ok else "missing_prediction",
                    "error_message": None if parsed_ok else "prediction row was missing",
                }
            )
    return pd.DataFrame(rows)


def _run_framework_predictions(
    cfg: ExperimentConfig,
    *,
    run_root: Path,
    system: EvalSystemConfig,
    instances: pd.DataFrame,
    index: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence_lookup = {str(row["instance_id"]): row for row in index.get("records", []) if isinstance(row, dict)}
    jobs = instances.to_dict(orient="records")
    results = run_jobs(
        jobs,
        worker=lambda job: _run_framework_for_instance(
            cfg,
            run_root=run_root,
            system=system,
            instance=job,
            evidence_record=evidence_lookup.get(str(job["instance_id"])),
            backends=backends,
            client=client,
        ),
        concurrency=int(system.concurrency),
    )
    pred_rows = [item[0] for item in results]
    trace_rows = [trace for _, traces in results for trace in traces]
    return pd.DataFrame(pred_rows), pd.DataFrame(trace_rows)


def _run_framework_for_instance(
    cfg: ExperimentConfig,
    *,
    run_root: Path,
    system: EvalSystemConfig,
    instance: dict[str, Any],
    evidence_record: dict[str, Any] | None,
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence = _read_evidence_payload(run_root, instance=instance, evidence_record=evidence_record)
    if system.framework_type == "single_llm":
        return _execute_single_llm(cfg, system=system, evidence=evidence, backends=backends, client=client)
    if system.framework_type == "healthcareagent":
        return _execute_healthcareagent(cfg, system=system, evidence=evidence, backends=backends, client=client)
    if system.framework_type == "reconcile":
        return _execute_reconcile(cfg, system=system, evidence=evidence, backends=backends, client=client)
    if system.framework_type == "mac":
        return _execute_mac(cfg, system=system, evidence=evidence, backends=backends, client=client)
    if system.framework_type == "medagent":
        return _execute_medagent(cfg, system=system, evidence=evidence, backends=backends, client=client)
    if system.framework_type == "colacare":
        return _execute_colacare(cfg, system=system, evidence=evidence, backends=backends, client=client)
    if system.framework_type == "mdagents":
        return _execute_mdagents(cfg, system=system, evidence=evidence, backends=backends, client=client)
    raise ValueError(f"Unsupported framework_type={system.framework_type!r}")


def _execute_single_llm(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    backend = _resolve_backend(system, backends, role_key="default", fallback_index=0)
    prompt = _render_prediction_prompt(cfg, evidence)
    stage = _run_prediction_stage(
        cfg,
        system=system,
        evidence=evidence,
        backend=backend,
        client=client,
        round_index=1,
        stage_name="decision",
        role_name="single_llm",
        prompt=prompt,
    )
    final_row = _finalize_prediction_row(
        cfg,
        system=system,
        evidence=evidence,
        parsed=stage.parsed if isinstance(stage.parsed, ParsedPrediction) else None,
        traces=[stage.trace_row],
    )
    return final_row, [stage.trace_row]


def _execute_healthcareagent(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    plan_backend = _resolve_backend(system, backends, role_key="planner", fallback_index=0)
    analysis_backend = _resolve_backend(system, backends, role_key="analysis", fallback_index=0)
    safety_backend = _resolve_backend(system, backends, role_key="safety", fallback_index=0)
    decision_backend = _resolve_backend(system, backends, role_key="decision", fallback_index=0)

    plan_prompt = (
        "Choose the safest first step for the medical prediction task.\n"
        "Return JSON only with {'answer': 'DIAGNOSE'|'INQUIRY'}.\n\n"
        f"{_render_evidence_context(evidence)}"
    )
    plan_stage = _run_json_stage(
        system=system,
        evidence=evidence,
        backend=plan_backend,
        client=client,
        round_index=1,
        stage_name="plan",
        role_name="planner",
        prompt=plan_prompt,
    )
    traces.append(plan_stage.trace_row)
    plan_payload = _stage_payload(plan_stage)
    needs_inquiry = str((plan_payload or {}).get("answer", "")).upper() == "INQUIRY"

    inquiry_questions: list[str] = []
    if needs_inquiry:
        inquiry_prompt = (
            "Generate up to three critical follow-up questions that would reduce uncertainty.\n"
            "Return JSON only with {'questions': ['...']}.\n\n"
            f"{_render_evidence_context(evidence)}"
        )
        inquiry_stage = _run_json_stage(
            system=system,
            evidence=evidence,
            backend=plan_backend,
            client=client,
            round_index=1,
            stage_name="inquiry",
            role_name="planner",
            prompt=inquiry_prompt,
        )
        traces.append(inquiry_stage.trace_row)
        inquiry_questions = [str(item) for item in ((_stage_payload(inquiry_stage) or {}).get("questions") or [])[:3]]

    inquiry_text = ""
    if inquiry_questions:
        inquiry_text = "Potential follow-up questions:\n" + "\n".join(f"- {item}" for item in inquiry_questions)
    analysis_prompt = (
        "Produce a preliminary prediction grounded only in the observed EHR evidence.\n"
        f"{inquiry_text}\n\n"
        f"{_render_prediction_prompt(cfg, evidence)}"
    )
    preliminary = _run_prediction_stage(
        cfg,
        system=system,
        evidence=evidence,
        backend=analysis_backend,
        client=client,
        round_index=1,
        stage_name="preliminary_analysis",
        role_name="preliminary_analyzer",
        prompt=analysis_prompt,
    )
    traces.append(preliminary.trace_row)

    safety_feedback: list[str] = []
    for stage_name, critique in [
        ("safety_ethics", "Check whether the draft overstates certainty or needs safety disclaimers."),
        ("safety_emergency", "Check whether the draft misses escalation or urgent risk signals."),
        ("safety_error", "Check for factual inconsistencies or unsupported claims in the draft."),
    ]:
        prompt = (
            f"{critique}\n"
            "Return JSON only with {'feedback': '...'}.\n\n"
            f"Draft prediction:\n{_prediction_brief(preliminary.parsed)}\n\n"
            f"{_render_evidence_context(evidence)}"
        )
        stage = _run_json_stage(
            system=system,
            evidence=evidence,
            backend=safety_backend,
            client=client,
            round_index=1,
            stage_name=stage_name,
            role_name=stage_name,
            prompt=prompt,
        )
        traces.append(stage.trace_row)
        feedback = str((_stage_payload(stage) or {}).get("feedback", "")).strip()
        if feedback:
            safety_feedback.append(f"{stage_name}: {feedback}")

    decision_prompt = (
        "Revise the draft prediction using the safety feedback below.\n"
        + ("\n".join(f"- {item}" for item in safety_feedback) if safety_feedback else "- No safety issues were found.")
        + "\n\n"
        + _render_prediction_prompt(cfg, evidence)
    )
    decision = _run_prediction_stage(
        cfg,
        system=system,
        evidence=evidence,
        backend=decision_backend,
        client=client,
        round_index=1,
        stage_name="final_decision",
        role_name="decision_maker",
        prompt=decision_prompt,
    )
    traces.append(decision.trace_row)
    final_row = _finalize_prediction_row(
        cfg,
        system=system,
        evidence=evidence,
        parsed=decision.parsed if isinstance(decision.parsed, ParsedPrediction) else None,
        traces=traces,
        metadata={"branch": "inquiry" if needs_inquiry else "diagnose"},
    )
    return final_row, traces


def _execute_reconcile(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    num_agents = int(system.framework_params.get("num_agents", max(len(system.backend_refs), 3)))
    round_outputs: list[list[ParsedPrediction | None]] = []
    latest_outputs: list[ParsedPrediction | None] = []

    for round_index in range(1, int(system.max_rounds) + 1):
        discussion_text = _discussion_summary(cfg, latest_outputs)
        current_round: list[ParsedPrediction | None] = []
        for agent_index in range(num_agents):
            backend = _resolve_backend(system, backends, role_key=f"agent_{agent_index + 1}", fallback_index=agent_index)
            prompt = (
                "You are a clinician participating in a multi-agent reconciliation discussion.\n"
                + (f"Prior discussion summary:\n{discussion_text}\n\n" if discussion_text else "")
                + _render_prediction_prompt(cfg, evidence)
            )
            stage = _run_prediction_stage(
                cfg,
                system=system,
                evidence=evidence,
                backend=backend,
                client=client,
                round_index=round_index,
                stage_name="discussion",
                role_name=f"agent_{agent_index + 1}",
                prompt=prompt,
            )
            traces.append(stage.trace_row)
            current_round.append(stage.parsed if isinstance(stage.parsed, ParsedPrediction) else None)
        round_outputs.append(current_round)
        latest_outputs = current_round
        if _has_consensus(cfg, latest_outputs):
            break

    final_parsed = _aggregate_peer_predictions(cfg, latest_outputs)
    traces.append(
        _synthetic_trace_row(
            system=system,
            evidence=evidence,
            round_index=len(round_outputs) or 1,
            stage_name="weighted_vote",
            role_name="coordinator",
            output_json={"consensus_reached": _has_consensus(cfg, latest_outputs), "prediction": _prediction_brief(final_parsed)},
        )
    )
    final_row = _finalize_prediction_row(
        cfg,
        system=system,
        evidence=evidence,
        parsed=final_parsed,
        traces=traces,
        metadata={"consensus_reached": _has_consensus(cfg, latest_outputs), "rounds_completed": len(round_outputs)},
    )
    return final_row, traces


def _execute_mac(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    num_doctors = int(system.framework_params.get("num_doctors", max(len(system.backend_refs), 4)))
    latest_doctors: list[ParsedPrediction | None] = []
    final_stage: StageExecution | None = None
    for round_index in range(1, int(system.max_rounds) + 1):
        doctor_outputs: list[ParsedPrediction | None] = []
        history_text = _discussion_summary(cfg, latest_doctors)
        for doctor_index in range(num_doctors):
            backend = _resolve_backend(system, backends, role_key=f"doctor_{doctor_index + 1}", fallback_index=doctor_index)
            prompt = (
                "You are a doctor in a supervised medical discussion.\n"
                + (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
                + _render_prediction_prompt(cfg, evidence)
            )
            stage = _run_prediction_stage(
                cfg,
                system=system,
                evidence=evidence,
                backend=backend,
                client=client,
                round_index=round_index,
                stage_name="doctor_analysis",
                role_name=f"doctor_{doctor_index + 1}",
                prompt=prompt,
            )
            traces.append(stage.trace_row)
            doctor_outputs.append(stage.parsed if isinstance(stage.parsed, ParsedPrediction) else None)
        latest_doctors = doctor_outputs
        supervisor = _resolve_backend(system, backends, role_key="supervisor", fallback_index=num_doctors)
        supervisor_prompt = (
            "You are the supervisor of a medical multi-agent discussion.\n"
            f"Doctor opinions:\n{_discussion_summary(cfg, doctor_outputs)}\n\n"
            + _render_prediction_prompt(cfg, evidence)
        )
        final_stage = _run_prediction_stage(
            cfg,
            system=system,
            evidence=evidence,
            backend=supervisor,
            client=client,
            round_index=round_index,
            stage_name="supervisor_decision",
            role_name="supervisor",
            prompt=supervisor_prompt,
        )
        traces.append(final_stage.trace_row)
        if _has_consensus(cfg, doctor_outputs):
            break

    final_row = _finalize_prediction_row(
        cfg,
        system=system,
        evidence=evidence,
        parsed=final_stage.parsed if final_stage and isinstance(final_stage.parsed, ParsedPrediction) else None,
        traces=traces,
        metadata={"consensus_reached": _has_consensus(cfg, latest_doctors)},
    )
    return final_row, traces


def _execute_medagent(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _execute_mdt_like(cfg, system=system, evidence=evidence, backends=backends, client=client, recruit_specialties=True, label="medagent")


def _execute_colacare(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _execute_mdt_like(cfg, system=system, evidence=evidence, backends=backends, client=client, recruit_specialties=False, label="colacare")


def _execute_mdagents(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    router = _resolve_backend(system, backends, role_key="router", fallback_index=0)
    router_prompt = (
        "Classify the case complexity.\n"
        "Return JSON only with {'complexity': 'basic'|'intermediate'|'advanced'}.\n\n"
        f"{_render_evidence_context(evidence)}"
    )
    router_stage = _run_json_stage(
        system=system,
        evidence=evidence,
        backend=router,
        client=client,
        round_index=1,
        stage_name="complexity_routing",
        role_name="router",
        prompt=router_prompt,
    )
    traces.append(router_stage.trace_row)
    complexity = str((_stage_payload(router_stage) or {}).get("complexity", "intermediate")).lower()
    if complexity not in {"basic", "intermediate", "advanced"}:
        complexity = "intermediate"

    if complexity == "basic":
        row, child_traces = _execute_single_llm(cfg, system=system, evidence=evidence, backends=backends, client=client)
    elif complexity == "intermediate":
        row, child_traces = _execute_mdt_like(cfg, system=system, evidence=evidence, backends=backends, client=client, recruit_specialties=True, label="mdagents_intermediate")
    else:
        row, child_traces = _execute_advanced_mdagents(cfg, system=system, evidence=evidence, backends=backends, client=client)
    traces.extend(child_traces)
    row["framework_metadata_json"] = json.dumps({"complexity": complexity}, ensure_ascii=False)
    return row, traces


def _execute_advanced_mdagents(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    team_count = int(system.framework_params.get("num_teams_advanced", 2))
    members_per_team = int(system.framework_params.get("num_agents_per_team_advanced", 2))
    team_reports: list[str] = []
    for team_index in range(team_count):
        member_outputs: list[ParsedPrediction | None] = []
        for member_index in range(members_per_team):
            backend = _resolve_backend(system, backends, role_key=f"team_{team_index + 1}_member_{member_index + 1}", fallback_index=member_index)
            prompt = (
                f"You are part of team {team_index + 1} in a hierarchical medical decision framework.\n"
                + _render_prediction_prompt(cfg, evidence)
            )
            stage = _run_prediction_stage(
                cfg,
                system=system,
                evidence=evidence,
                backend=backend,
                client=client,
                round_index=1,
                stage_name="team_member_analysis",
                role_name=f"team_{team_index + 1}_member_{member_index + 1}",
                prompt=prompt,
            )
            traces.append(stage.trace_row)
            member_outputs.append(stage.parsed if isinstance(stage.parsed, ParsedPrediction) else None)
        lead_backend = _resolve_backend(system, backends, role_key=f"team_{team_index + 1}_lead", fallback_index=team_index)
        lead_prompt = (
            f"Synthesize the reports from team {team_index + 1}.\n"
            f"Team reports:\n{_discussion_summary(cfg, member_outputs)}\n\n"
            + _render_prediction_prompt(cfg, evidence)
        )
        lead_stage = _run_prediction_stage(
            cfg,
            system=system,
            evidence=evidence,
            backend=lead_backend,
            client=client,
            round_index=1,
            stage_name="team_synthesis",
            role_name=f"team_{team_index + 1}_lead",
            prompt=lead_prompt,
        )
        traces.append(lead_stage.trace_row)
        team_reports.append(_prediction_brief(lead_stage.parsed))

    decision_backend = _resolve_backend(system, backends, role_key="decision", fallback_index=0)
    decision_prompt = (
        "Make the final decision using the team lead reports below.\n"
        + "\n".join(f"- {item}" for item in team_reports)
        + "\n\n"
        + _render_prediction_prompt(cfg, evidence)
    )
    decision = _run_prediction_stage(
        cfg,
        system=system,
        evidence=evidence,
        backend=decision_backend,
        client=client,
        round_index=1,
        stage_name="final_decision",
        role_name="decision_maker",
        prompt=decision_prompt,
    )
    traces.append(decision.trace_row)
    row = _finalize_prediction_row(
        cfg,
        system=system,
        evidence=evidence,
        parsed=decision.parsed if isinstance(decision.parsed, ParsedPrediction) else None,
        traces=traces,
        metadata={"team_count": team_count, "members_per_team": members_per_team},
    )
    return row, traces


def _execute_mdt_like(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backends: dict[str, EvalBackendConfig],
    client: OpenAICompatibleAgentClient,
    recruit_specialties: bool,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    specialties = DEFAULT_SPECIALTIES[: max(3, len(system.backend_refs) or 3)]
    if recruit_specialties:
        recruiter_backend = _resolve_backend(system, backends, role_key="recruiter", fallback_index=0)
        recruiter_prompt = (
            "Choose the most relevant specialties for this case.\n"
            "Return JSON only with {'specialties': ['...']}.\n\n"
            f"{_render_evidence_context(evidence)}"
        )
        recruiter_stage = _run_json_stage(
            system=system,
            evidence=evidence,
            backend=recruiter_backend,
            client=client,
            round_index=1,
            stage_name="role_recruitment",
            role_name="recruiter",
            prompt=recruiter_prompt,
        )
        traces.append(recruiter_stage.trace_row)
        recruited = [str(item) for item in ((_stage_payload(recruiter_stage) or {}).get("specialties") or []) if str(item).strip()]
        if recruited:
            specialties = recruited[: max(3, len(recruited))]

    round_outputs: list[ParsedPrediction | None] = []
    for round_index in range(1, int(system.max_rounds) + 1):
        doctor_outputs: list[ParsedPrediction | None] = []
        prior = _discussion_summary(cfg, round_outputs)
        for doctor_index, specialty in enumerate(specialties):
            backend = _resolve_backend(system, backends, role_key=f"doctor_{doctor_index + 1}", fallback_index=doctor_index)
            prompt = (
                f"You are the {specialty} specialist in a multidisciplinary consultation.\n"
                + (f"Prior round summary:\n{prior}\n\n" if prior else "")
                + _render_prediction_prompt(cfg, evidence)
            )
            stage = _run_prediction_stage(
                cfg,
                system=system,
                evidence=evidence,
                backend=backend,
                client=client,
                round_index=round_index,
                stage_name="doctor_analysis",
                role_name=f"doctor_{doctor_index + 1}",
                prompt=prompt,
            )
            traces.append(stage.trace_row)
            doctor_outputs.append(stage.parsed if isinstance(stage.parsed, ParsedPrediction) else None)

        meta_backend = _resolve_backend(system, backends, role_key="meta", fallback_index=0)
        synthesis_prompt = (
            "Synthesize the specialist reports into a coherent consensus draft.\n"
            f"Reports:\n{_discussion_summary(cfg, doctor_outputs)}\n\n"
            + _render_prediction_prompt(cfg, evidence)
        )
        synthesis = _run_prediction_stage(
            cfg,
            system=system,
            evidence=evidence,
            backend=meta_backend,
            client=client,
            round_index=round_index,
            stage_name="synthesis",
            role_name="meta",
            prompt=synthesis_prompt,
        )
        traces.append(synthesis.trace_row)

        review_flags: list[bool] = []
        for doctor_index, specialty in enumerate(specialties):
            backend = _resolve_backend(system, backends, role_key=f"reviewer_{doctor_index + 1}", fallback_index=doctor_index)
            review_prompt = (
                f"You are the {specialty} specialist reviewing the synthesis.\n"
                "Return JSON only with {'agree': true|false, 'reason': '...'}.\n\n"
                f"Synthesis:\n{_prediction_brief(synthesis.parsed)}\n\n"
                f"{_render_evidence_context(evidence)}"
            )
            review = _run_json_stage(
                system=system,
                evidence=evidence,
                backend=backend,
                client=client,
                round_index=round_index,
                stage_name="review",
                role_name=f"reviewer_{doctor_index + 1}",
                prompt=review_prompt,
            )
            traces.append(review.trace_row)
            review_flags.append(bool((_stage_payload(review) or {}).get("agree")))

        decision_backend = _resolve_backend(system, backends, role_key="decision", fallback_index=0)
        decision_prompt = (
            "Make the final decision from the synthesis and specialist reviews.\n"
            f"Synthesis:\n{_prediction_brief(synthesis.parsed)}\n"
            f"Review agreement count: {sum(1 for flag in review_flags if flag)}/{len(review_flags)}\n\n"
            + _render_prediction_prompt(cfg, evidence)
        )
        decision = _run_prediction_stage(
            cfg,
            system=system,
            evidence=evidence,
            backend=decision_backend,
            client=client,
            round_index=round_index,
            stage_name="decision",
            role_name="decision_maker",
            prompt=decision_prompt,
        )
        traces.append(decision.trace_row)
        round_outputs = doctor_outputs
        if all(review_flags) or round_index >= int(system.max_rounds):
            row = _finalize_prediction_row(
                cfg,
                system=system,
                evidence=evidence,
                parsed=decision.parsed if isinstance(decision.parsed, ParsedPrediction) else None,
                traces=traces,
                metadata={"framework_label": label, "specialties": specialties, "consensus_reached": all(review_flags)},
            )
            return row, traces

    row = _finalize_prediction_row(
        cfg,
        system=system,
        evidence=evidence,
        parsed=None,
        traces=traces,
        metadata={"framework_label": label, "specialties": specialties, "consensus_reached": False},
        error_code="decision_missing",
        error_message="framework finished without a final decision",
    )
    return row, traces


def _run_prediction_stage(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backend: EvalBackendConfig,
    client: OpenAICompatibleAgentClient,
    round_index: int,
    stage_name: str,
    role_name: str,
    prompt: str,
) -> StageExecution:
    response_format = build_response_format(
        task_kind=cfg.task.kind,
        include_explanation=True,
        include_confidence=True,
        supports_json_schema=backend.supports_json_schema,
    )
    request = AgentRequestSpec(
        backend_name=backend.name,
        provider_model=backend.model,
        base_url=backend.base_url,
        api_key_env=backend.api_key_env,
        prompt=prompt,
        system_prompt=backend.system_prompt,
        response_format=response_format,
        temperature=float(system.temperature),
        top_p=float(system.top_p),
        timeout_seconds=float(system.timeout_seconds),
        max_retries=int(system.max_retries),
        seed=system.seed,
        extra_headers=dict(backend.headers),
    )
    execution = execute_agent_request(
        client=client,
        request=request,
        parse_response=lambda raw_text: parse_prediction_response(
            raw_text,
            task_kind=cfg.task.kind,
            include_explanation=True,
            include_confidence=True,
        ),
    )
    parsed = execution.parsed
    row = _base_trace_row(
        system=system,
        evidence=evidence,
        backend=backend,
        round_index=round_index,
        stage_name=stage_name,
        role_name=role_name,
        prompt=prompt,
        raw_response=execution.raw_response,
        prompt_sha256=execution.prompt_sha256,
        response_sha256=execution.response_sha256,
        parsed_ok=getattr(parsed, "parsed_ok", False),
        stage_output_json=_prediction_json(parsed if isinstance(parsed, ParsedPrediction) else None),
        latency_ms=execution.latency_ms,
        token_usage_prompt=execution.token_usage_prompt,
        token_usage_completion=execution.token_usage_completion,
        token_usage_total=execution.token_usage_total,
        cost_usd=_compute_cost(
            backend,
            prompt_tokens=execution.token_usage_prompt,
            completion_tokens=execution.token_usage_completion,
        ),
        error_code=execution.error_code,
        error_message=execution.error_message,
    )
    return StageExecution(parsed=parsed, trace_row=row)


def _run_json_stage(
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backend: EvalBackendConfig,
    client: OpenAICompatibleAgentClient,
    round_index: int,
    stage_name: str,
    role_name: str,
    prompt: str,
) -> StageExecution:
    request = AgentRequestSpec(
        backend_name=backend.name,
        provider_model=backend.model,
        base_url=backend.base_url,
        api_key_env=backend.api_key_env,
        prompt=prompt,
        system_prompt=backend.system_prompt,
        response_format={"type": "json_object"},
        temperature=float(system.temperature),
        top_p=float(system.top_p),
        timeout_seconds=float(system.timeout_seconds),
        max_retries=int(system.max_retries),
        seed=system.seed,
        extra_headers=dict(backend.headers),
    )
    execution = execute_agent_request(client=client, request=request, parse_response=_parse_json_stage)
    parsed = execution.parsed
    row = _base_trace_row(
        system=system,
        evidence=evidence,
        backend=backend,
        round_index=round_index,
        stage_name=stage_name,
        role_name=role_name,
        prompt=prompt,
        raw_response=execution.raw_response,
        prompt_sha256=execution.prompt_sha256,
        response_sha256=execution.response_sha256,
        parsed_ok=getattr(parsed, "parsed_ok", False),
        stage_output_json=json.dumps(as_jsonable((parsed.payload if isinstance(parsed, ParsedStageJson) and parsed.payload is not None else {})), ensure_ascii=False),
        latency_ms=execution.latency_ms,
        token_usage_prompt=execution.token_usage_prompt,
        token_usage_completion=execution.token_usage_completion,
        token_usage_total=execution.token_usage_total,
        cost_usd=_compute_cost(
            backend,
            prompt_tokens=execution.token_usage_prompt,
            completion_tokens=execution.token_usage_completion,
        ),
        error_code=execution.error_code,
        error_message=execution.error_message,
    )
    return StageExecution(parsed=parsed, trace_row=row)


def _finalize_prediction_row(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    parsed: ParsedPrediction | None,
    traces: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    token_prompt = int(sum(int(row.get("token_usage_prompt") or 0) for row in traces))
    token_completion = int(sum(int(row.get("token_usage_completion") or 0) for row in traces))
    token_total = int(sum(int(row.get("token_usage_total") or 0) for row in traces))
    cost_total = float(sum(float(row.get("cost_usd") or 0.0) for row in traces))
    latency_total = int(sum(int(row.get("latency_ms") or 0) for row in traces if row.get("latency_ms") is not None))
    parsed_ok = bool(parsed is not None and parsed.parsed_ok)
    label = int(parsed.prediction) if parsed_ok and cfg.task.kind == "binary" else None
    probability = float(parsed.probability) if parsed_ok and cfg.task.kind == "binary" and parsed.probability is not None else None
    value = float(parsed.value) if parsed_ok and cfg.task.kind == "regression" and parsed.value is not None else None
    return {
        "instance_id": str(evidence["instance_id"]),
        "patient_id": str(evidence["patient_id"]),
        "split": str(evidence["split"]),
        "split_role": str(evidence.get("split_role", "test")),
        "bin_time": evidence.get("bin_time"),
        "ground_truth": evidence.get("ground_truth"),
        "system_name": system.name,
        "system_kind": system.kind,
        "framework_type": system.framework_type,
        "parsed_ok": parsed_ok,
        "prediction": label if cfg.task.kind == "binary" else value,
        "label": label,
        "probability": probability,
        "value": value,
        "explanation": None if parsed is None else parsed.explanation,
        "confidence": None if parsed is None else parsed.confidence,
        "latency_ms": latency_total,
        "token_usage_prompt": token_prompt,
        "token_usage_completion": token_completion,
        "token_usage_total": token_total,
        "cost_usd": cost_total,
        "round_count": int(max((int(row.get("round") or 0) for row in traces), default=0)),
        "trace_row_count": int(len(traces)),
        "config_sha256": _system_config_sha256(system),
        "framework_metadata_json": json.dumps(as_jsonable(metadata or {}), ensure_ascii=False),
        "error_code": error_code or (None if parsed_ok else (None if parsed is None else parsed.error_code)),
        "error_message": error_message or (None if parsed_ok else (None if parsed is None else parsed.error_message)),
    }


def _empty_prediction_row(
    cfg: ExperimentConfig,
    *,
    system: EvalSystemConfig,
    instance: dict[str, Any],
    error_code: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "instance_id": str(instance["instance_id"]),
        "patient_id": str(instance["patient_id"]),
        "split": str(instance["split"]),
        "split_role": str(instance.get("split_role", "test")),
        "bin_time": _json_time(instance.get("bin_time")),
        "ground_truth": instance.get("ground_truth"),
        "system_name": system.name,
        "system_kind": system.kind,
        "framework_type": system.framework_type,
        "parsed_ok": False,
        "prediction": None,
        "label": None,
        "probability": None,
        "value": None,
        "explanation": None,
        "confidence": None,
        "latency_ms": None,
        "token_usage_prompt": 0,
        "token_usage_completion": 0,
        "token_usage_total": 0,
        "cost_usd": 0.0,
        "round_count": 0,
        "trace_row_count": 0,
        "config_sha256": _system_config_sha256(system),
        "framework_metadata_json": json.dumps({}, ensure_ascii=False),
        "error_code": error_code,
        "error_message": error_message,
    }


def _base_trace_row(
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    backend: EvalBackendConfig,
    round_index: int,
    stage_name: str,
    role_name: str,
    prompt: str,
    raw_response: str,
    prompt_sha256: str,
    response_sha256: str | None,
    parsed_ok: bool,
    stage_output_json: str,
    latency_ms: int | None,
    token_usage_prompt: int | None,
    token_usage_completion: int | None,
    token_usage_total: int | None,
    cost_usd: float | None,
    error_code: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        "instance_id": str(evidence["instance_id"]),
        "patient_id": str(evidence["patient_id"]),
        "split": str(evidence["split"]),
        "bin_time": evidence.get("bin_time"),
        "system_name": system.name,
        "framework_type": system.framework_type,
        "backend_name": backend.name,
        "provider_model": backend.model,
        "round": int(round_index),
        "stage": str(stage_name),
        "role": str(role_name),
        "actor_name": str(role_name),
        "parsed_ok": bool(parsed_ok),
        "prompt": prompt,
        "raw_response": raw_response,
        "prompt_sha256": prompt_sha256,
        "response_sha256": response_sha256,
        "stage_output_json": stage_output_json,
        "latency_ms": latency_ms,
        "token_usage_prompt": token_usage_prompt,
        "token_usage_completion": token_usage_completion,
        "token_usage_total": token_usage_total,
        "cost_usd": cost_usd,
        "error_code": error_code,
        "error_message": error_message,
        "config_sha256": _system_config_sha256(system),
    }


def _synthetic_trace_row(
    *,
    system: EvalSystemConfig,
    evidence: dict[str, Any],
    round_index: int,
    stage_name: str,
    role_name: str,
    output_json: dict[str, Any],
) -> dict[str, Any]:
    return {
        "instance_id": str(evidence["instance_id"]),
        "patient_id": str(evidence["patient_id"]),
        "split": str(evidence["split"]),
        "bin_time": evidence.get("bin_time"),
        "system_name": system.name,
        "framework_type": system.framework_type,
        "backend_name": None,
        "provider_model": None,
        "round": int(round_index),
        "stage": str(stage_name),
        "role": str(role_name),
        "actor_name": str(role_name),
        "parsed_ok": True,
        "prompt": None,
        "raw_response": None,
        "prompt_sha256": None,
        "response_sha256": None,
        "stage_output_json": json.dumps(as_jsonable(output_json), ensure_ascii=False),
        "latency_ms": None,
        "token_usage_prompt": 0,
        "token_usage_completion": 0,
        "token_usage_total": 0,
        "cost_usd": 0.0,
        "error_code": None,
        "error_message": None,
        "config_sha256": _system_config_sha256(system),
    }


def _score_prediction_frame(frame: pd.DataFrame, *, task_kind: str) -> dict[str, Any]:
    if frame.empty:
        return {"coverage": 0.0, "scored_rows": 0, "metrics": {}}
    work = frame.copy()
    work["ground_truth"] = pd.to_numeric(work["ground_truth"], errors="coerce")
    work = work[work["parsed_ok"] == True].copy()  # noqa: E712
    if task_kind == "binary":
        work["probability"] = pd.to_numeric(work["probability"], errors="coerce")
        scored = work.dropna(subset=["ground_truth", "probability"]).copy()
        if scored.empty:
            return {"coverage": float((frame["parsed_ok"] == True).mean()), "scored_rows": 0, "metrics": {}}  # noqa: E712
        metrics = binary_metrics(
            scored["ground_truth"].to_numpy(dtype=float),
            scored["probability"].to_numpy(dtype=float),
        ).metrics
    else:
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        scored = work.dropna(subset=["ground_truth", "value"]).copy()
        if scored.empty:
            return {"coverage": float((frame["parsed_ok"] == True).mean()), "scored_rows": 0, "metrics": {}}  # noqa: E712
        metrics = regression_metrics(
            scored["ground_truth"].to_numpy(dtype=float),
            scored["value"].to_numpy(dtype=float),
        ).metrics
    return {
        "coverage": float((frame["parsed_ok"] == True).mean()),  # noqa: E712
        "scored_rows": int(len(scored)),
        "metrics": {str(key): float(value) for key, value in metrics.items()},
    }


def _build_pairwise_rows(
    cfg: ExperimentConfig,
    *,
    prediction_frames: dict[str, pd.DataFrame],
    primary_metric: str,
) -> list[dict[str, Any]]:
    suites = cfg.eval.suites or [
        type(
            "Suite",
            (),
            {
                "name": "default",
                "compare_pairs": [(left, right) for left in prediction_frames for right in prediction_frames if left < right],
                "include_systems": list(prediction_frames),
                "min_coverage": 0.0,
            },
        )()
    ]

    rows: list[dict[str, Any]] = []
    for suite in suites:
        for left_name, right_name in suite.compare_pairs:
            left = prediction_frames.get(left_name, pd.DataFrame())
            right = prediction_frames.get(right_name, pd.DataFrame())
            if left.empty or right.empty:
                continue
            shared = left.merge(
                right,
                on="instance_id",
                how="inner",
                suffixes=("_left", "_right"),
            )
            if shared.empty:
                continue
            if cfg.task.kind == "binary":
                shared["ground_truth_left"] = pd.to_numeric(shared["ground_truth_left"], errors="coerce")
                shared["probability_left"] = pd.to_numeric(shared["probability_left"], errors="coerce")
                shared["probability_right"] = pd.to_numeric(shared["probability_right"], errors="coerce")
                shared = shared[
                    (shared["parsed_ok_left"] == True)  # noqa: E712
                    & (shared["parsed_ok_right"] == True)  # noqa: E712
                ].dropna(subset=["ground_truth_left", "probability_left", "probability_right"])
                if shared.empty:
                    continue
                left_metrics = binary_metrics(shared["ground_truth_left"].to_numpy(dtype=float), shared["probability_left"].to_numpy(dtype=float)).metrics
                right_metrics = binary_metrics(shared["ground_truth_left"].to_numpy(dtype=float), shared["probability_right"].to_numpy(dtype=float)).metrics
                delta_ci = _bootstrap_metric_delta(
                    y_true=shared["ground_truth_left"].to_numpy(dtype=float),
                    left_pred=shared["probability_left"].to_numpy(dtype=float),
                    right_pred=shared["probability_right"].to_numpy(dtype=float),
                    task_kind=cfg.task.kind,
                    metric=primary_metric,
                    n=cfg.eval.bootstrap_samples,
                    seed=cfg.eval.seed,
                )
            else:
                shared["ground_truth_left"] = pd.to_numeric(shared["ground_truth_left"], errors="coerce")
                shared["value_left"] = pd.to_numeric(shared["value_left"], errors="coerce")
                shared["value_right"] = pd.to_numeric(shared["value_right"], errors="coerce")
                shared = shared[
                    (shared["parsed_ok_left"] == True)  # noqa: E712
                    & (shared["parsed_ok_right"] == True)  # noqa: E712
                ].dropna(subset=["ground_truth_left", "value_left", "value_right"])
                if shared.empty:
                    continue
                left_metrics = regression_metrics(shared["ground_truth_left"].to_numpy(dtype=float), shared["value_left"].to_numpy(dtype=float)).metrics
                right_metrics = regression_metrics(shared["ground_truth_left"].to_numpy(dtype=float), shared["value_right"].to_numpy(dtype=float)).metrics
                delta_ci = _bootstrap_metric_delta(
                    y_true=shared["ground_truth_left"].to_numpy(dtype=float),
                    left_pred=shared["value_left"].to_numpy(dtype=float),
                    right_pred=shared["value_right"].to_numpy(dtype=float),
                    task_kind=cfg.task.kind,
                    metric=primary_metric,
                    n=cfg.eval.bootstrap_samples,
                    seed=cfg.eval.seed,
                )
            rows.append(
                {
                    "suite_name": suite.name,
                    "left_system": left_name,
                    "right_system": right_name,
                    "shared_rows": int(len(shared)),
                    "metric": primary_metric,
                    "left_value": float(left_metrics[primary_metric]),
                    "right_value": float(right_metrics[primary_metric]),
                    "delta": float(left_metrics[primary_metric] - right_metrics[primary_metric]),
                    "delta_ci_low": delta_ci["ci_low"],
                    "delta_ci_high": delta_ci["ci_high"],
                }
            )
    return rows


def _bootstrap_metric_delta(
    *,
    y_true: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    task_kind: str,
    metric: str,
    n: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    values: list[float] = []
    for _ in range(int(n)):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample]
        lp = left_pred[sample]
        rp = right_pred[sample]
        if task_kind == "binary":
            left_metric = binary_metrics(yt.astype(float), lp.astype(float)).metrics[metric]
            right_metric = binary_metrics(yt.astype(float), rp.astype(float)).metrics[metric]
        else:
            left_metric = regression_metrics(yt.astype(float), lp.astype(float)).metrics[metric]
            right_metric = regression_metrics(yt.astype(float), rp.astype(float)).metrics[metric]
        values.append(float(left_metric - right_metric))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def _read_evidence_payload(run_root: Path, *, instance: dict[str, Any], evidence_record: dict[str, Any] | None) -> dict[str, Any]:
    payload = {
        "instance_id": str(instance["instance_id"]),
        "patient_id": str(instance["patient_id"]),
        "split": str(instance["split"]),
        "split_role": str(instance.get("split_role", "test")),
        "prediction_mode": str(instance.get("prediction_mode", "patient")),
        "task_kind": str(instance.get("task_kind", "")),
        "bin_time": _json_time(instance.get("bin_time")),
        "ground_truth": instance.get("ground_truth"),
        "events": [],
        "static_features": {},
        "analysis_refs": {"modules": [], "patient_case_matches": []},
    }
    if evidence_record is None or not evidence_record.get("evidence_path"):
        return payload

    evidence_path = run_root / str(evidence_record["evidence_path"])
    evidence_meta = json.loads(evidence_path.read_text(encoding="utf-8"))
    artifacts = evidence_meta.get("artifacts") or {}
    events_path = run_root / str(artifacts.get("events_csv"))
    static_path = run_root / str(artifacts.get("static_json"))
    refs_path = run_root / str(artifacts.get("analysis_refs_json"))
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    static_payload = json.loads(static_path.read_text(encoding="utf-8")) if static_path.exists() else {"features": {}}
    refs_payload = json.loads(refs_path.read_text(encoding="utf-8")) if refs_path.exists() else {"modules": [], "patient_case_matches": []}
    payload["events"] = events.to_dict(orient="records")
    payload["static_features"] = {str(key): value for key, value in (static_payload.get("features") or {}).items()}
    payload["analysis_refs"] = refs_payload
    return payload


def _collect_analysis_refs(*, run_root: Path, patient_id: str) -> dict[str, Any]:
    try:
        index = read_analysis_index(run_root)
    except FileNotFoundError:
        index = {"modules": []}
    matches: list[dict[str, Any]] = []
    for module_name in ("prediction_audit",):
        try:
            desc = describe_patient_case(run_root, patient_id, module_name=module_name)
        except FileNotFoundError:
            continue
        for item in list(desc.get("matches", [])):
            if isinstance(item, dict):
                matches.append(
                    {
                        "module": module_name,
                        "name": str(item.get("split", "case")),
                        "row_count": 1,
                        "record": item,
                    }
                )
    return {
        "modules": list(index.get("modules", [])) if isinstance(index, dict) else [],
        "patient_case_matches": matches,
    }


def _load_dynamic_by_patient(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    if dynamic is None or dynamic.empty:
        return {}
    tmp = dynamic.copy()
    tmp["patient_id"] = tmp["patient_id"].astype(str)
    tmp["event_time"] = pd.to_datetime(tmp["event_time"], errors="raise")
    return {
        str(pid): group.sort_values("event_time", kind="stable").reset_index(drop=True)
        for pid, group in tmp.groupby("patient_id", sort=False)
    }


def _load_static_by_patient(cfg: ExperimentConfig) -> dict[str, pd.Series]:
    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    static = load_static_table(train_dataset.static)
    if static is None or static.empty:
        return {}
    tmp = static.copy()
    tmp["patient_id"] = tmp["patient_id"].astype(str)
    return {
        str(row["patient_id"]): row
        for _, row in tmp.drop_duplicates(subset=["patient_id"], keep="last").iterrows()
    }


def _resolve_backend(
    system: EvalSystemConfig,
    backends: dict[str, EvalBackendConfig],
    *,
    role_key: str,
    fallback_index: int,
) -> EvalBackendConfig:
    direct_key = system.framework_params.get(f"{role_key}_backend_ref")
    if isinstance(direct_key, str) and direct_key in backends:
        return backends[direct_key]
    if system.backend_refs:
        if fallback_index < len(system.backend_refs):
            return backends[system.backend_refs[fallback_index]]
        return backends[system.backend_refs[0]]
    raise ValueError(f"System {system.name!r} has no backend_refs configured")


def _render_prediction_prompt(cfg: ExperimentConfig, evidence: dict[str, Any]) -> str:
    schema_text = schema_prompt_text(
        task_kind=cfg.task.kind,
        include_explanation=True,
        include_confidence=True,
    )
    return (
        f"{_render_evidence_context(evidence)}\n\n"
        "Task:\n"
        + (
            "- Predict the binary label as 0 or 1 and provide probability in [0, 1].\n"
            if cfg.task.kind == "binary"
            else "- Predict the target numeric value.\n"
        )
        + "Ground your answer only in the observed evidence above.\n"
        + "Return exactly one JSON object matching this schema:\n"
        + schema_text
    )


def _render_evidence_context(evidence: dict[str, Any]) -> str:
    lines = [
        "Case Context",
        f"- instance_id: {evidence['instance_id']}",
        f"- patient_id: {evidence['patient_id']}",
        f"- split: {evidence['split']}",
        f"- prediction_mode: {evidence.get('prediction_mode')}",
    ]
    if evidence.get("bin_time") not in {None, "", "NaT"}:
        lines.append(f"- anchor_time: {evidence['bin_time']}")
    lines.append("")
    lines.append("Static Features")
    static_features = evidence.get("static_features") or {}
    if static_features:
        for key, value in static_features.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Observed Timeline")
    events = list(evidence.get("events") or [])
    if not events:
        lines.append("- no events available")
    else:
        for item in events:
            lines.append(f"- {item.get('event_time')} | {item.get('code')} = {item.get('value')}")
    return "\n".join(lines).strip()


def _stage_payload(stage: StageExecution) -> dict[str, Any] | None:
    if isinstance(stage.parsed, ParsedStageJson):
        return stage.parsed.payload
    return None


def _parse_json_stage(text: str) -> ParsedStageJson:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return ParsedStageJson(parsed_ok=False, error_code="invalid_json", error_message="response did not contain a JSON object")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParsedStageJson(parsed_ok=False, error_code="invalid_json", error_message=str(exc))
    if not isinstance(payload, dict):
        return ParsedStageJson(parsed_ok=False, error_code="invalid_payload", error_message="JSON response must be an object")
    return ParsedStageJson(parsed_ok=True, payload=payload)


def _extract_json_candidate(text: str) -> str | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        for part in stripped.split("```"):
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def _has_consensus(cfg: ExperimentConfig, outputs: list[ParsedPrediction | None]) -> bool:
    valid = [item for item in outputs if item is not None and item.parsed_ok]
    if not valid:
        return False
    if cfg.task.kind == "binary":
        labels = {int(item.prediction) for item in valid if item.prediction is not None}
        return len(labels) == 1
    values = [float(item.value) for item in valid if item.value is not None]
    return bool(values) and float(np.std(values)) <= 1e-6


def _aggregate_peer_predictions(cfg: ExperimentConfig, outputs: list[ParsedPrediction | None]) -> ParsedPrediction | None:
    valid = [item for item in outputs if item is not None and item.parsed_ok]
    if not valid:
        return None
    weights = np.asarray([_confidence_weight(item.confidence) for item in valid], dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(valid), dtype=float) / len(valid)
    if cfg.task.kind == "binary":
        probs = np.asarray([
            float(item.probability if item.probability is not None else float(item.prediction))
            for item in valid
        ], dtype=float)
        probability = float(np.dot(weights, probs))
        label = int(probability >= 0.5)
        explanations = [item.explanation for item in valid if item.explanation]
        return ParsedPrediction(
            parsed_ok=True,
            prediction=label,
            probability=probability,
            explanation=" | ".join(explanations[:3]) or None,
            confidence=float(np.mean([item.confidence for item in valid if item.confidence is not None])) if any(item.confidence is not None for item in valid) else None,
        )
    values = np.asarray([float(item.value) for item in valid if item.value is not None], dtype=float)
    if values.size == 0:
        return None
    value = float(np.dot(weights[: values.size], values))
    explanations = [item.explanation for item in valid if item.explanation]
    return ParsedPrediction(
        parsed_ok=True,
        prediction=value,
        value=value,
        explanation=" | ".join(explanations[:3]) or None,
        confidence=float(np.mean([item.confidence for item in valid if item.confidence is not None])) if any(item.confidence is not None for item in valid) else None,
    )


def _discussion_summary(cfg: ExperimentConfig, outputs: list[ParsedPrediction | None]) -> str:
    valid = [item for item in outputs if item is not None and item.parsed_ok]
    if not valid:
        return ""
    lines = []
    for idx, item in enumerate(valid, start=1):
        if cfg.task.kind == "binary":
            lines.append(f"- clinician_{idx}: label={item.prediction} probability={item.probability} confidence={item.confidence} explanation={item.explanation}")
        else:
            lines.append(f"- clinician_{idx}: value={item.value} confidence={item.confidence} explanation={item.explanation}")
    return "\n".join(lines)


def _prediction_brief(parsed: ParsedPrediction | None) -> str:
    if parsed is None or not parsed.parsed_ok:
        return "prediction unavailable"
    payload = _prediction_json(parsed)
    return json.dumps(json.loads(payload), ensure_ascii=False)


def _prediction_json(parsed: ParsedPrediction | None) -> str:
    if parsed is None or not parsed.parsed_ok:
        return json.dumps({}, ensure_ascii=False)
    payload: dict[str, Any] = {}
    if parsed.prediction is not None:
        if parsed.probability is not None:
            payload["label"] = int(parsed.prediction)
            payload["probability"] = float(parsed.probability)
        elif parsed.value is not None:
            payload["value"] = float(parsed.value)
    if parsed.explanation is not None:
        payload["explanation"] = parsed.explanation
    if parsed.confidence is not None:
        payload["confidence"] = float(parsed.confidence)
    return json.dumps(as_jsonable(payload), ensure_ascii=False)


def _confidence_weight(confidence: float | None) -> float:
    if confidence is None:
        return 1.0
    return max(0.1, min(1.0, float(confidence)))


def _compute_cost(
    backend: EvalBackendConfig,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    if backend.prompt_token_cost_per_1k is None and backend.completion_token_cost_per_1k is None:
        return None
    total = 0.0
    if backend.prompt_token_cost_per_1k is not None and prompt_tokens is not None:
        total += (float(prompt_tokens) / 1000.0) * float(backend.prompt_token_cost_per_1k)
    if backend.completion_token_cost_per_1k is not None and completion_tokens is not None:
        total += (float(completion_tokens) / 1000.0) * float(backend.completion_token_cost_per_1k)
    return total


def _system_config_sha256(system: EvalSystemConfig) -> str:
    payload = json.dumps(as_jsonable(asdict(system)), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value in {None, "", "NaT"}:
        return None
    return pd.to_datetime(value, errors="raise")


def _json_time(value: object) -> str | None:
    if value in {None, "", "NaT"}:
        return None
    return pd.to_datetime(value, errors="raise").isoformat()


def _series_to_dict(row: pd.Series | None) -> dict[str, Any]:
    if row is None or row.empty:
        return {}
    out: dict[str, Any] = {}
    for key, value in row.items():
        if str(key) == "patient_id" or pd.isna(value):
            continue
        out[str(key)] = value
    return as_jsonable(out)


def _instance_dir_name(instance_id: str) -> str:
    base = safe_case_slug(instance_id)[:80]
    suffix = hashlib.sha1(str(instance_id).encode("utf-8")).hexdigest()[:10]
    return f"{base}_{suffix}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_eval_index(cfg: ExperimentConfig, *, run_root: Path) -> dict[str, Any]:
    index_path = run_root / "eval" / "index.json"
    if not index_path.exists():
        build_eval_artifacts(cfg, run_root=run_root, force=False)
    return _read_json(index_path)


def _ensure_eval_summary(cfg: ExperimentConfig, *, run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "eval" / "summary.json"
    if not summary_path.exists():
        run_eval_systems(cfg, run_root=run_root, force=False)
    return _read_json(summary_path)
