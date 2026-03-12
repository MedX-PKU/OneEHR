from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.agent import (
    AgentClientError,
    AgentRequestSpec,
    OpenAICompatibleAgentClient,
    render_prompt,
    safe_case_slug,
    validate_agent_review_setup,
)
from oneehr.agent.instances import (
    agent_instance_path,
    materialize_agent_instances,
    validate_agent_predict_setup,
)
from oneehr.agent.predict_eval import summarize_prediction_rows
from oneehr.agent.predict_schema import (
    build_response_format,
    parse_prediction_response,
    schema_prompt_text,
)
from oneehr.agent.review_eval import summarize_review_rows
from oneehr.agent.review_schema import (
    build_review_response_format,
    parse_review_response,
    review_schema_prompt_text,
)
from oneehr.cases import list_cases, materialize_cases, read_case
from oneehr.cli._common import resolve_run_root
from oneehr.utils.io import ensure_dir, write_json, write_jsonl


def register_agent_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("agent", help="Run agent prediction and review workflows")
    agent_sub = parser.add_subparsers(dest="agent_command")

    predict = agent_sub.add_parser("predict", help="Run agent prediction over materialized instances")
    predict.add_argument("--config", required=True, help="Path to TOML config")
    predict.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    predict.add_argument("--force", action="store_true", help="Overwrite existing agent prediction artifacts")
    predict.set_defaults(handler=run_agent_predict)

    review = agent_sub.add_parser("review", help="Run agent review over case predictions")
    review.add_argument("--config", required=True, help="Path to TOML config")
    review.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    review.add_argument("--force", action="store_true", help="Overwrite existing agent review artifacts")
    review.set_defaults(handler=run_agent_review)


def run_agent_predict(args: argparse.Namespace) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.data.io import load_dynamic_table_optional, load_static_table

    cfg = load_experiment_config(args.config)
    validate_agent_predict_setup(cfg)
    run_root = resolve_run_root(cfg, args.run_dir)
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}. Run `oneehr preprocess` first.")

    predict_root = ensure_dir(run_root / "agent" / "predict")
    if args.force:
        for rel in ["prompts", "responses", "parsed", "preds", "metrics", "failures", "summary.json"]:
            path = predict_root / rel
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()

    instance_path = agent_instance_path(run_root, cfg.agent.predict.sample_unit)
    if not instance_path.exists():
        materialize_agent_instances(cfg, run_root=run_root)
    instances = pd.read_parquet(instance_path)
    if instances.empty:
        write_json(predict_root / "summary.json", {"records": []})
        return

    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static = load_static_table(train_dataset.static) if cfg.agent.predict.prompt.include_static else None
    dynamic_by_patient = _group_dynamic(dynamic)
    static_by_patient = _group_static(static)

    client = OpenAICompatibleAgentClient()
    schema_text = schema_prompt_text(
        task_kind=cfg.task.kind,
        include_explanation=cfg.agent.predict.output.include_explanation,
        include_confidence=cfg.agent.predict.output.include_confidence,
    )

    records: list[dict[str, object]] = []
    for backend in cfg.agent.predict.backends:
        frame = _run_predict_backend(
            cfg=cfg,
            predictor_name=backend.name,
            provider_model=backend.model,
            client=client,
            instances=instances,
            dynamic_by_patient=dynamic_by_patient,
            static_by_patient=static_by_patient,
            schema_text=schema_text,
            base_url=backend.base_url,
            api_key_env=backend.api_key_env,
            system_prompt=backend.system_prompt,
            supports_json_schema=backend.supports_json_schema,
            headers=backend.headers,
        )

        for split_name, split_df in frame.groupby("split", sort=True):
            split_df = split_df.sort_values(["patient_id", "instance_id"], kind="stable").reset_index(drop=True)
            prompt_rows = split_df[
                ["instance_id", "patient_id", "split", "split_role", "predictor_name", "prompt", "prompt_sha256"]
            ].to_dict(orient="records")
            if "bin_time" in split_df.columns:
                for idx, row in enumerate(prompt_rows):
                    bin_time = split_df.iloc[idx]["bin_time"]
                    if pd.notna(bin_time):
                        row["bin_time"] = pd.to_datetime(bin_time).isoformat()

            response_rows = split_df[
                [
                    "instance_id",
                    "patient_id",
                    "split",
                    "split_role",
                    "predictor_name",
                    "raw_response",
                    "response_sha256",
                    "latency_ms",
                    "token_usage_prompt",
                    "token_usage_completion",
                    "token_usage_total",
                    "error_code",
                    "error_message",
                ]
            ].to_dict(orient="records")
            if "bin_time" in split_df.columns:
                for idx, row in enumerate(response_rows):
                    bin_time = split_df.iloc[idx]["bin_time"]
                    if pd.notna(bin_time):
                        row["bin_time"] = pd.to_datetime(bin_time).isoformat()

            prompt_path = None
            response_path = None
            if cfg.agent.predict.save_prompts:
                prompt_path = ensure_dir(predict_root / "prompts" / backend.name) / f"{split_name}.jsonl"
                write_jsonl(prompt_path, prompt_rows)
            if cfg.agent.predict.save_responses:
                response_path = ensure_dir(predict_root / "responses" / backend.name) / f"{split_name}.jsonl"
                write_jsonl(response_path, response_rows)

            if cfg.agent.predict.save_parsed:
                parsed_path = ensure_dir(predict_root / "parsed" / backend.name) / f"{split_name}.parquet"
                parsed_cols = [
                    "instance_id",
                    "patient_id",
                    "split",
                    "split_role",
                    "predictor_name",
                    "ground_truth",
                    "parsed_ok",
                    "prediction",
                    "probability",
                    "value",
                    "explanation",
                    "confidence",
                    "error_code",
                    "error_message",
                ]
                if "bin_time" in split_df.columns:
                    parsed_cols.insert(2, "bin_time")
                split_df[parsed_cols].to_parquet(parsed_path, index=False)

            pred_path = ensure_dir(predict_root / "preds" / backend.name) / f"{split_name}.parquet"
            pred_cols = [
                "instance_id",
                "patient_id",
                "split",
                "split_role",
                "predictor_name",
                "ground_truth",
                "parsed_ok",
                "prediction",
                "probability",
                "value",
                "explanation",
                "confidence",
                "prompt_sha256",
                "response_sha256",
                "token_usage_prompt",
                "token_usage_completion",
                "token_usage_total",
                "latency_ms",
                "error_code",
                "error_message",
            ]
            if "bin_time" in split_df.columns:
                pred_cols.insert(2, "bin_time")
            split_df[pred_cols].to_parquet(pred_path, index=False)

            failures = split_df[split_df["parsed_ok"] == False].copy()  # noqa: E712
            if not failures.empty:
                failure_path = ensure_dir(predict_root / "failures" / backend.name) / f"{split_name}.jsonl"
                write_jsonl(
                    failure_path,
                    failures[
                        [
                            "instance_id",
                            "patient_id",
                            "split",
                            "split_role",
                            "predictor_name",
                            "error_code",
                            "error_message",
                            "raw_response",
                        ]
                    ].to_dict(orient="records"),
                )

            metrics = summarize_prediction_rows(split_df, task_kind=cfg.task.kind)
            metrics_path = ensure_dir(predict_root / "metrics" / backend.name) / f"{split_name}.json"
            write_json(metrics_path, metrics)
            records.append(
                {
                    "predictor_name": backend.name,
                    "provider_model": backend.model,
                    "split": split_name,
                    "task_kind": cfg.task.kind,
                    "prediction_mode": cfg.task.prediction_mode,
                    "metrics": metrics.get("metrics", {}),
                    "total_rows": metrics["total_rows"],
                    "parsed_ok_rows": metrics["parsed_ok_rows"],
                    "parse_success_rate": metrics["parse_success_rate"],
                    "ground_truth_rows": metrics["ground_truth_rows"],
                    "scored_rows": metrics["scored_rows"],
                    "coverage": metrics["coverage"],
                    "artifacts": {
                        "metrics_json": str(metrics_path.relative_to(run_root)),
                        "preds_parquet": str(pred_path.relative_to(run_root)),
                        "prompts_jsonl": None if prompt_path is None else str(prompt_path.relative_to(run_root)),
                        "responses_jsonl": None if response_path is None else str(response_path.relative_to(run_root)),
                    },
                }
            )

    write_json(
        predict_root / "summary.json",
        {
            "run_name": str(cfg.output.run_name),
            "task": {"kind": str(cfg.task.kind), "prediction_mode": str(cfg.task.prediction_mode)},
            "agent_predict": {
                "sample_unit": str(cfg.agent.predict.sample_unit),
                "prompt_template": str(cfg.agent.predict.prompt_template),
                "json_schema_version": int(cfg.agent.predict.json_schema_version),
            },
            "records": records,
        },
    )


def run_agent_review(args: argparse.Namespace) -> None:
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(args.config)
    validate_agent_review_setup(cfg)
    run_root = resolve_run_root(cfg, args.run_dir)
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}. Run `oneehr preprocess` first.")

    if not (run_root / "cases" / "index.json").exists():
        materialize_cases(cfg, run_root=run_root, force=False)

    review_root = ensure_dir(run_root / "agent" / "review")
    if args.force:
        for rel in ["prompts", "responses", "parsed", "metrics", "failures", "summary.json"]:
            path = review_root / rel
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()

    case_summaries = list_cases(run_root, limit=cfg.agent.review.max_cases)
    if not case_summaries:
        write_json(review_root / "summary.json", {"records": []})
        return

    client = OpenAICompatibleAgentClient()
    schema_text = review_schema_prompt_text()
    records: list[dict[str, Any]] = []
    for reviewer in cfg.agent.review.backends:
        frame = _run_review_backend(
            cfg=cfg,
            run_root=run_root,
            reviewer=reviewer,
            client=client,
            case_summaries=case_summaries,
            schema_text=schema_text,
        )
        if frame.empty:
            continue

        for split_name, split_df in frame.groupby("split", sort=True):
            split_df = split_df.sort_values(
                ["case_id", "target_origin", "target_predictor_name"],
                kind="stable",
            ).reset_index(drop=True)

            prompt_path = None
            response_path = None
            parsed_path = None
            if cfg.agent.review.save_prompts:
                prompt_path = ensure_dir(review_root / "prompts" / reviewer.name) / f"{split_name}.jsonl"
                write_jsonl(
                    prompt_path,
                    split_df[
                        [
                            "review_id",
                            "case_id",
                            "patient_id",
                            "split",
                            "target_origin",
                            "target_predictor_name",
                            "prompt",
                            "prompt_sha256",
                        ]
                    ].to_dict(orient="records"),
                )
            if cfg.agent.review.save_responses:
                response_path = ensure_dir(review_root / "responses" / reviewer.name) / f"{split_name}.jsonl"
                write_jsonl(
                    response_path,
                    split_df[
                        [
                            "review_id",
                            "case_id",
                            "patient_id",
                            "split",
                            "target_origin",
                            "target_predictor_name",
                            "raw_response",
                            "response_sha256",
                            "latency_ms",
                            "token_usage_prompt",
                            "token_usage_completion",
                            "token_usage_total",
                            "error_code",
                            "error_message",
                        ]
                    ].to_dict(orient="records"),
                )
            if cfg.agent.review.save_parsed:
                parsed_path = ensure_dir(review_root / "parsed" / reviewer.name) / f"{split_name}.parquet"
                split_df[
                    [
                        "review_id",
                        "case_id",
                        "patient_id",
                        "split",
                        "target_origin",
                        "target_predictor_name",
                        "parsed_ok",
                        "supported",
                        "clinically_grounded",
                        "leakage_suspected",
                        "needs_human_review",
                        "overall_score",
                        "review_summary",
                        "key_evidence_json",
                        "missing_evidence_json",
                        "ground_truth",
                        "error_code",
                        "error_message",
                    ]
                ].to_parquet(parsed_path, index=False)

            failures = split_df[split_df["parsed_ok"] == False].copy()  # noqa: E712
            if not failures.empty:
                failure_path = ensure_dir(review_root / "failures" / reviewer.name) / f"{split_name}.jsonl"
                write_jsonl(
                    failure_path,
                    failures[
                        [
                            "review_id",
                            "case_id",
                            "patient_id",
                            "split",
                            "target_origin",
                            "target_predictor_name",
                            "error_code",
                            "error_message",
                            "raw_response",
                        ]
                    ].to_dict(orient="records"),
                )

            for (target_origin, target_predictor_name), block in split_df.groupby(
                ["target_origin", "target_predictor_name"],
                sort=True,
            ):
                metrics = summarize_review_rows(block)
                metric_name = (
                    f"{split_name}__{safe_case_slug(str(target_origin))}"
                    f"__{safe_case_slug(str(target_predictor_name))}"
                )
                metrics_path = ensure_dir(review_root / "metrics" / reviewer.name) / f"{metric_name}.json"
                write_json(metrics_path, metrics)
                records.append(
                    {
                        "reviewer_name": reviewer.name,
                        "provider_model": reviewer.model,
                        "split": split_name,
                        "target_origin": str(target_origin),
                        "target_predictor_name": str(target_predictor_name),
                        "metrics": metrics.get("metrics", {}),
                        "total_rows": metrics["total_rows"],
                        "parsed_ok_rows": metrics["parsed_ok_rows"],
                        "parse_success_rate": metrics["parse_success_rate"],
                        "artifacts": {
                            "metrics_json": str(metrics_path.relative_to(run_root)),
                            "parsed_parquet": None if parsed_path is None else str(parsed_path.relative_to(run_root)),
                            "prompts_jsonl": None if prompt_path is None else str(prompt_path.relative_to(run_root)),
                            "responses_jsonl": None if response_path is None else str(response_path.relative_to(run_root)),
                        },
                    }
                )

    write_json(
        review_root / "summary.json",
        {
            "run_name": str(cfg.output.run_name),
            "task": {"kind": str(cfg.task.kind), "prediction_mode": str(cfg.task.prediction_mode)},
            "agent_review": {
                "prompt_template": str(cfg.agent.review.prompt_template),
                "json_schema_version": int(cfg.agent.review.json_schema_version),
                "prediction_origins": list(cfg.agent.review.prediction_origins),
            },
            "records": records,
        },
    )


def _run_predict_backend(
    *,
    cfg,
    predictor_name: str,
    provider_model: str,
    client,
    instances: pd.DataFrame,
    dynamic_by_patient: dict[str, pd.DataFrame],
    static_by_patient: dict[str, pd.Series],
    schema_text: str,
    base_url: str,
    api_key_env: str,
    system_prompt: str | None,
    supports_json_schema: bool,
    headers: dict[str, str],
) -> pd.DataFrame:
    response_format = build_response_format(
        task_kind=cfg.task.kind,
        include_explanation=cfg.agent.predict.output.include_explanation,
        include_confidence=cfg.agent.predict.output.include_confidence,
        supports_json_schema=supports_json_schema,
    )

    jobs = instances.to_dict(orient="records")
    if cfg.agent.predict.concurrency <= 1:
        rows = [
            _predict_one(
                cfg=cfg,
                instance=job,
                predictor_name=predictor_name,
                provider_model=provider_model,
                client=client,
                dynamic_by_patient=dynamic_by_patient,
                static_by_patient=static_by_patient,
                schema_text=schema_text,
                response_format=response_format,
                base_url=base_url,
                api_key_env=api_key_env,
                system_prompt=system_prompt,
                headers=headers,
            )
            for job in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=int(cfg.agent.predict.concurrency)) as ex:
            rows = list(
                ex.map(
                    lambda job: _predict_one(
                        cfg=cfg,
                        instance=job,
                        predictor_name=predictor_name,
                        provider_model=provider_model,
                        client=client,
                        dynamic_by_patient=dynamic_by_patient,
                        static_by_patient=static_by_patient,
                        schema_text=schema_text,
                        response_format=response_format,
                        base_url=base_url,
                        api_key_env=api_key_env,
                        system_prompt=system_prompt,
                        headers=headers,
                    ),
                    jobs,
                )
            )
    return pd.DataFrame(rows)


def _predict_one(
    *,
    cfg,
    instance: dict[str, object],
    predictor_name: str,
    provider_model: str,
    client,
    dynamic_by_patient: dict[str, pd.DataFrame],
    static_by_patient: dict[str, pd.Series],
    schema_text: str,
    response_format: dict[str, object],
    base_url: str,
    api_key_env: str,
    system_prompt: str | None,
    headers: dict[str, str],
) -> dict[str, object]:
    patient_id = str(instance["patient_id"])
    dynamic = dynamic_by_patient.get(patient_id)
    static_row = static_by_patient.get(patient_id)
    prompt = render_prompt(
        cfg=cfg,
        instance=instance,
        dynamic=dynamic,
        static_row=static_row,
        schema_text=schema_text,
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    request = AgentRequestSpec(
        backend_name=predictor_name,
        provider_model=provider_model,
        base_url=base_url,
        api_key_env=api_key_env,
        prompt=prompt,
        system_prompt=system_prompt,
        response_format=response_format,
        temperature=cfg.agent.predict.temperature,
        top_p=cfg.agent.predict.top_p,
        timeout_seconds=cfg.agent.predict.timeout_seconds,
        max_retries=cfg.agent.predict.max_retries,
        seed=cfg.agent.predict.seed,
        extra_headers=dict(headers),
    )

    raw_text = ""
    latency_ms = None
    usage_prompt = None
    usage_completion = None
    usage_total = None
    error_code = None
    error_message = None
    try:
        response = client.complete(request)
        raw_text = response.raw_text
        latency_ms = response.latency_ms
        usage_prompt = response.usage_prompt_tokens
        usage_completion = response.usage_completion_tokens
        usage_total = response.usage_total_tokens
        parsed = parse_prediction_response(
            raw_text,
            task_kind=cfg.task.kind,
            include_explanation=cfg.agent.predict.output.include_explanation,
            include_confidence=cfg.agent.predict.output.include_confidence,
        )
    except AgentClientError as exc:
        error_code = exc.code
        error_message = exc.message
        raw_text = exc.response_text or ""
        parsed = (
            parse_prediction_response(
                raw_text,
                task_kind=cfg.task.kind,
                include_explanation=cfg.agent.predict.output.include_explanation,
                include_confidence=cfg.agent.predict.output.include_confidence,
            )
            if raw_text
            else None
        )

    if parsed is None:
        parsed_ok = False
        prediction = None
        probability = None
        value = None
        explanation = None
        confidence = None
        if error_code is None:
            error_code = "request_failed"
            error_message = "request failed before a response was returned"
    else:
        parsed_ok = parsed.parsed_ok
        prediction = parsed.prediction
        probability = parsed.probability
        value = parsed.value
        explanation = parsed.explanation
        confidence = parsed.confidence
        if not parsed.parsed_ok and error_code is None:
            error_code = parsed.error_code
            error_message = parsed.error_message

    response_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else None
    row = {
        "instance_id": instance["instance_id"],
        "patient_id": patient_id,
        "split": instance["split"],
        "split_role": instance.get("split_role", "test"),
        "ground_truth": instance.get("ground_truth"),
        "parsed_ok": parsed_ok,
        "prediction": prediction,
        "probability": probability,
        "value": value,
        "explanation": explanation,
        "confidence": confidence,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "raw_response": raw_text,
        "response_sha256": response_sha256,
        "token_usage_prompt": usage_prompt,
        "token_usage_completion": usage_completion,
        "token_usage_total": usage_total,
        "latency_ms": latency_ms,
        "error_code": error_code,
        "error_message": error_message,
        "predictor_name": predictor_name,
        "provider_model": provider_model,
    }
    if "bin_time" in instance:
        row["bin_time"] = pd.to_datetime(instance["bin_time"], errors="raise")
    return row
def _run_review_backend(
    *,
    cfg,
    run_root: Path,
    reviewer,
    client,
    case_summaries: list[dict[str, Any]],
    schema_text: str,
) -> pd.DataFrame:
    response_format = build_review_response_format(supports_json_schema=reviewer.supports_json_schema)
    jobs = _build_review_jobs(
        run_root=run_root,
        case_summaries=case_summaries,
        allowed_origins=set(cfg.agent.review.prediction_origins),
    )
    if not jobs:
        return pd.DataFrame()

    if cfg.agent.review.concurrency <= 1:
        rows = [
            _review_one(
                cfg=cfg,
                job=job,
                reviewer=reviewer,
                client=client,
                schema_text=schema_text,
                response_format=response_format,
            )
            for job in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=int(cfg.agent.review.concurrency)) as ex:
            rows = list(
                ex.map(
                    lambda job: _review_one(
                        cfg=cfg,
                        job=job,
                        reviewer=reviewer,
                        client=client,
                        schema_text=schema_text,
                        response_format=response_format,
                    ),
                    jobs,
                )
            )
    return pd.DataFrame(rows)


def _build_review_jobs(
    *,
    run_root: Path,
    case_summaries: list[dict[str, Any]],
    allowed_origins: set[str],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for summary in case_summaries:
        case_id = str(summary["case_id"])
        case = read_case(run_root, case_id)
        preds = pd.DataFrame(case.get("predictions", []))
        if preds.empty:
            continue
        preds = preds[preds["origin"].astype(str).isin(sorted(allowed_origins))].copy()
        if preds.empty:
            continue
        for _, pred in preds.iterrows():
            jobs.append({"case": case, "target_prediction": pred.to_dict()})
    return jobs


def _review_one(
    *,
    cfg,
    job: dict[str, Any],
    reviewer,
    client,
    schema_text: str,
    response_format: dict[str, object],
) -> dict[str, Any]:
    case = dict(job["case"])
    target_prediction = dict(job["target_prediction"])
    case_id = str(case["case_id"])
    patient_id = str(case["patient_id"])

    dynamic = pd.DataFrame(case.get("events", []))
    static_payload = case.get("static", {})
    static_features = {}
    if isinstance(static_payload, dict):
        maybe = static_payload.get("features", {})
        if isinstance(maybe, dict):
            static_features = maybe
    static_row = pd.Series({"patient_id": patient_id, **static_features}) if static_features else None
    instance = {
        "case_id": case_id,
        "patient_id": patient_id,
        "split": case["split"],
        "split_role": case.get("split_role", "test"),
        "bin_time": case.get("bin_time"),
        "ground_truth": case.get("ground_truth") if cfg.agent.review.prompt.include_ground_truth else None,
        "prediction_mode": case.get("prediction_mode", cfg.task.prediction_mode),
    }

    prompt = render_prompt(
        cfg=cfg,
        instance=instance,
        dynamic=dynamic,
        static_row=static_row,
        schema_text=schema_text,
        template_name=cfg.agent.review.prompt_template,
        prompt_cfg=cfg.agent.review.prompt,
        target_prediction=target_prediction,
        analysis_refs=(case.get("analysis_refs") if cfg.agent.review.prompt.include_analysis_context else None),
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    review_id = (
        f"{case_id}|{target_prediction.get('origin')}|{target_prediction.get('predictor_name')}"
    )
    request = AgentRequestSpec(
        backend_name=reviewer.name,
        provider_model=reviewer.model,
        base_url=reviewer.base_url,
        api_key_env=reviewer.api_key_env,
        prompt=prompt,
        system_prompt=reviewer.system_prompt,
        response_format=response_format,
        temperature=cfg.agent.review.temperature,
        top_p=cfg.agent.review.top_p,
        timeout_seconds=cfg.agent.review.timeout_seconds,
        max_retries=cfg.agent.review.max_retries,
        seed=cfg.agent.review.seed,
        extra_headers=dict(reviewer.headers),
    )

    raw_text = ""
    latency_ms = None
    usage_prompt = None
    usage_completion = None
    usage_total = None
    error_code = None
    error_message = None
    try:
        response = client.complete(request)
        raw_text = response.raw_text
        latency_ms = response.latency_ms
        usage_prompt = response.usage_prompt_tokens
        usage_completion = response.usage_completion_tokens
        usage_total = response.usage_total_tokens
        parsed = parse_review_response(raw_text)
    except AgentClientError as exc:
        error_code = exc.code
        error_message = exc.message
        raw_text = exc.response_text or ""
        parsed = parse_review_response(raw_text) if raw_text else None

    if parsed is None:
        parsed_ok = False
        supported = None
        clinically_grounded = None
        leakage_suspected = None
        needs_human_review = None
        overall_score = None
        review_summary = None
        key_evidence_json = "[]"
        missing_evidence_json = "[]"
        if error_code is None:
            error_code = "request_failed"
            error_message = "request failed before a response was returned"
    else:
        parsed_ok = parsed.parsed_ok
        supported = parsed.supported
        clinically_grounded = parsed.clinically_grounded
        leakage_suspected = parsed.leakage_suspected
        needs_human_review = parsed.needs_human_review
        overall_score = parsed.overall_score
        review_summary = parsed.review_summary
        key_evidence_json = json.dumps(parsed.key_evidence, ensure_ascii=False)
        missing_evidence_json = json.dumps(parsed.missing_evidence, ensure_ascii=False)
        if not parsed.parsed_ok and error_code is None:
            error_code = parsed.error_code
            error_message = parsed.error_message

    response_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else None
    row = {
        "review_id": review_id,
        "case_id": case_id,
        "patient_id": patient_id,
        "split": case["split"],
        "target_origin": target_prediction.get("origin"),
        "target_predictor_name": target_prediction.get("predictor_name"),
        "ground_truth": case.get("ground_truth"),
        "parsed_ok": parsed_ok,
        "supported": supported,
        "clinically_grounded": clinically_grounded,
        "leakage_suspected": leakage_suspected,
        "needs_human_review": needs_human_review,
        "overall_score": overall_score,
        "review_summary": review_summary,
        "key_evidence_json": key_evidence_json,
        "missing_evidence_json": missing_evidence_json,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "raw_response": raw_text,
        "response_sha256": response_sha256,
        "token_usage_prompt": usage_prompt,
        "token_usage_completion": usage_completion,
        "token_usage_total": usage_total,
        "latency_ms": latency_ms,
        "error_code": error_code,
        "error_message": error_message,
        "reviewer_name": reviewer.name,
    }
    if case.get("bin_time") not in {None, "", "NaT"}:
        row["bin_time"] = pd.to_datetime(case["bin_time"], errors="raise")
    return row


def _group_dynamic(dynamic: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if dynamic is None or dynamic.empty:
        return {}
    tmp = dynamic.copy()
    tmp["patient_id"] = tmp["patient_id"].astype(str)
    tmp["event_time"] = pd.to_datetime(tmp["event_time"], errors="raise")
    return {
        str(pid): grp.sort_values("event_time", kind="stable").reset_index(drop=True)
        for pid, grp in tmp.groupby("patient_id", sort=False)
    }


def _group_static(static: pd.DataFrame | None) -> dict[str, pd.Series]:
    if static is None or static.empty:
        return {}
    tmp = static.copy()
    tmp["patient_id"] = tmp["patient_id"].astype(str)
    return {
        str(row["patient_id"]): row
        for _, row in tmp.drop_duplicates(subset=["patient_id"], keep="last").iterrows()
    }
