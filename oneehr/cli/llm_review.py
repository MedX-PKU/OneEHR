"""oneehr llm-review subcommand."""
from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.agent.workspace import list_workspace_cases, materialize_case_workspaces, read_workspace_case
from oneehr.cli._common import resolve_run_root
from oneehr.llm.client import LLMClientError, OpenAICompatibleChatClient
from oneehr.llm.contracts import LLMRequestSpec
from oneehr.llm.render import render_prompt
from oneehr.llm.templates import get_prompt_template, safe_case_slug
from oneehr.review.eval import summarize_review_rows
from oneehr.review.schema import (
    build_review_response_format,
    parse_review_response,
    review_schema_prompt_text,
)
from oneehr.utils.io import ensure_dir, write_json, write_jsonl


def run_llm_review(cfg_path: str, *, run_dir: str | None, force: bool) -> None:
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(cfg_path)
    _validate_review_setup(cfg)
    run_root = resolve_run_root(cfg, run_dir)
    if not run_root.exists():
        raise SystemExit(
            f"Run directory not found: {run_root}. "
            "Run `oneehr preprocess` first."
        )

    if not (run_root / "workspace" / "index.json").exists():
        materialize_case_workspaces(cfg, run_root=run_root, force=False)

    review_root = run_root / "review"
    ensure_dir(review_root)
    if force:
        for rel in ["prompts", "responses", "parsed", "metrics", "failures", "summary.json"]:
            path = review_root / rel
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()

    cases = list_workspace_cases(run_root, limit=cfg.review.max_cases)
    if not cases:
        write_json(review_root / "summary.json", {"records": []})
        return

    client = OpenAICompatibleChatClient()
    schema_text = review_schema_prompt_text()
    records: list[dict[str, Any]] = []
    for review_model in cfg.review_models:
        frame = _run_review_model(
            cfg=cfg,
            run_root=run_root,
            review_model=review_model,
            client=client,
            cases=cases,
            schema_text=schema_text,
        )
        if frame.empty:
            continue
        for split_name, split_df in frame.groupby("split", sort=True):
            split_df = split_df.sort_values(["case_id", "target_source", "target_model_name"], kind="stable").reset_index(drop=True)

            prompt_path = None
            response_path = None
            parsed_path = None
            if cfg.review.save_prompts:
                prompt_path = ensure_dir(review_root / "prompts" / review_model.name) / f"{split_name}.jsonl"
                write_jsonl(
                    prompt_path,
                    split_df[
                        [
                            "review_id",
                            "case_id",
                            "patient_id",
                            "split",
                            "target_source",
                            "target_model_name",
                            "prompt",
                            "prompt_sha256",
                        ]
                    ].to_dict(orient="records"),
                )
            if cfg.review.save_responses:
                response_path = ensure_dir(review_root / "responses" / review_model.name) / f"{split_name}.jsonl"
                write_jsonl(
                    response_path,
                    split_df[
                        [
                            "review_id",
                            "case_id",
                            "patient_id",
                            "split",
                            "target_source",
                            "target_model_name",
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
            if cfg.review.save_parsed:
                parsed_path = ensure_dir(review_root / "parsed" / review_model.name) / f"{split_name}.parquet"
                split_df[
                    [
                        "review_id",
                        "case_id",
                        "patient_id",
                        "split",
                        "target_source",
                        "target_model_name",
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
                failure_path = ensure_dir(review_root / "failures" / review_model.name) / f"{split_name}.jsonl"
                write_jsonl(
                    failure_path,
                    failures[
                        [
                            "review_id",
                            "case_id",
                            "patient_id",
                            "split",
                            "target_source",
                            "target_model_name",
                            "error_code",
                            "error_message",
                            "raw_response",
                        ]
                    ].to_dict(orient="records"),
                )

            for (target_source, target_model_name), block in split_df.groupby(["target_source", "target_model_name"], sort=True):
                metrics = summarize_review_rows(block)
                metric_name = f"{split_name}__{safe_case_slug(str(target_source))}__{safe_case_slug(str(target_model_name))}"
                metrics_path = ensure_dir(review_root / "metrics" / review_model.name) / f"{metric_name}.json"
                write_json(metrics_path, metrics)
                records.append(
                    {
                        "review_model": review_model.name,
                        "provider_model": review_model.model,
                        "split": split_name,
                        "target_source": str(target_source),
                        "target_model_name": str(target_model_name),
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
            "review": {
                "prompt_template": str(cfg.review.prompt_template),
                "json_schema_version": int(cfg.review.json_schema_version),
                "prediction_sources": list(cfg.review.prediction_sources),
            },
            "records": records,
        },
    )


def _validate_review_setup(cfg) -> None:
    if not cfg.review.enabled:
        raise SystemExit("Reviewer workflow is disabled. Set review.enabled = true in the config.")
    try:
        template = get_prompt_template(cfg.review.prompt_template)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    if template.family != "review":
        raise SystemExit(f"review.prompt_template must resolve to a review template, got {template.family!r}.")
    if cfg.task.kind not in set(template.supported_task_kinds):
        raise SystemExit(
            f"review.prompt_template={cfg.review.prompt_template!r} does not support task.kind={cfg.task.kind!r}."
        )
    if cfg.task.prediction_mode not in set(template.supported_sample_units):
        raise SystemExit(
            f"review.prompt_template={cfg.review.prompt_template!r} does not support "
            f"prediction_mode={cfg.task.prediction_mode!r}."
        )
    if cfg.review.json_schema_version not in set(template.supported_schema_versions):
        raise SystemExit(
            f"review.prompt_template={cfg.review.prompt_template!r} does not support "
            f"json_schema_version={cfg.review.json_schema_version!r}."
        )
    if cfg.review.prompt.include_ground_truth and not template.allow_labels_context:
        raise SystemExit(
            f"review.prompt.include_ground_truth is not allowed for prompt template {cfg.review.prompt_template!r}."
        )
    if not cfg.review_models:
        raise SystemExit("At least one [[review_models]] entry is required for the reviewer workflow.")


def _run_review_model(
    *,
    cfg,
    run_root: Path,
    review_model,
    client,
    cases: list[dict[str, Any]],
    schema_text: str,
) -> pd.DataFrame:
    response_format = build_review_response_format(supports_json_schema=review_model.supports_json_schema)
    jobs = _build_review_jobs(run_root=run_root, cases=cases, allowed_sources=set(cfg.review.prediction_sources))
    if not jobs:
        return pd.DataFrame()

    if cfg.review.concurrency <= 1:
        rows = [
            _review_one(
                cfg=cfg,
                job=job,
                review_model=review_model,
                client=client,
                schema_text=schema_text,
                response_format=response_format,
            )
            for job in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=int(cfg.review.concurrency)) as ex:
            rows = list(
                ex.map(
                    lambda job: _review_one(
                        cfg=cfg,
                        job=job,
                        review_model=review_model,
                        client=client,
                        schema_text=schema_text,
                        response_format=response_format,
                    ),
                    jobs,
                )
            )
    return pd.DataFrame(rows)


def _build_review_jobs(*, run_root: Path, cases: list[dict[str, Any]], allowed_sources: set[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for summary in cases:
        case_id = str(summary["case_id"])
        case = read_workspace_case(run_root, case_id)
        preds = pd.DataFrame(case.get("predictions", []))
        if preds.empty:
            continue
        preds = preds[preds["source"].astype(str).isin(sorted(allowed_sources))].copy()
        if preds.empty:
            continue
        for _, pred in preds.iterrows():
            jobs.append(
                {
                    "case": case,
                    "target_prediction": pred.to_dict(),
                }
            )
    return jobs


def _review_one(
    *,
    cfg,
    job: dict[str, Any],
    review_model,
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
        "ground_truth": case.get("ground_truth") if cfg.review.prompt.include_ground_truth else None,
        "prediction_mode": case.get("prediction_mode", cfg.task.prediction_mode),
    }

    prompt = render_prompt(
        cfg=cfg,
        instance=instance,
        dynamic=dynamic,
        static_row=static_row,
        schema_text=schema_text,
        template_name=cfg.review.prompt_template,
        prompt_cfg=cfg.review.prompt,
        target_prediction=target_prediction,
        analysis_refs=(case.get("analysis_refs") if cfg.review.prompt.include_analysis_context else None),
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    review_id = f"{case_id}|{target_prediction.get('source')}|{target_prediction.get('model_name')}"
    req = LLMRequestSpec(
        model_name=review_model.name,
        provider_model=review_model.model,
        base_url=review_model.base_url,
        api_key_env=review_model.api_key_env,
        prompt=prompt,
        system_prompt=review_model.system_prompt,
        response_format=response_format,
        temperature=cfg.review.temperature,
        top_p=cfg.review.top_p,
        timeout_seconds=cfg.review.timeout_seconds,
        max_retries=cfg.review.max_retries,
        seed=cfg.review.seed,
        extra_headers=dict(review_model.headers),
    )

    raw_text = ""
    latency_ms = None
    usage_prompt = None
    usage_completion = None
    usage_total = None
    error_code = None
    error_message = None
    try:
        resp = client.complete(req)
        raw_text = resp.raw_text
        latency_ms = resp.latency_ms
        usage_prompt = resp.usage_prompt_tokens
        usage_completion = resp.usage_completion_tokens
        usage_total = resp.usage_total_tokens
        parsed = parse_review_response(raw_text)
    except LLMClientError as exc:
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
        "target_source": target_prediction.get("source"),
        "target_model_name": target_prediction.get("model_name"),
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
        "review_model": review_model.name,
    }
    if case.get("bin_time") not in {None, "", "NaT"}:
        row["bin_time"] = pd.to_datetime(case["bin_time"], errors="raise")
    return row
