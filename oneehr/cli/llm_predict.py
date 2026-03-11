"""oneehr llm-predict subcommand."""
from __future__ import annotations

import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from oneehr.cli._common import resolve_run_root
from oneehr.llm.client import LLMClientError, OpenAICompatibleChatClient
from oneehr.llm.contracts import LLMRequestSpec
from oneehr.llm.eval import summarize_prediction_rows
from oneehr.llm.instances import llm_instance_path, materialize_llm_instances, validate_llm_setup
from oneehr.llm.render import render_prompt
from oneehr.llm.schema import build_response_format, parse_prediction_response, schema_prompt_text
from oneehr.utils.io import ensure_dir, write_json, write_jsonl


def run_llm_predict(cfg_path: str, *, run_dir: str | None, force: bool) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.data.io import load_dynamic_table_optional, load_static_table

    cfg = load_experiment_config(cfg_path)
    validate_llm_setup(cfg)
    run_root = resolve_run_root(cfg, run_dir)
    if not run_root.exists():
        raise SystemExit(
            f"Run directory not found: {run_root}. "
            "Run `oneehr preprocess` first."
        )

    llm_root = run_root / "llm"
    ensure_dir(llm_root)
    if force:
        for rel in ["prompts", "responses", "parsed", "preds", "metrics", "failures", "summary.json"]:
            path = llm_root / rel
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()

    instance_path = llm_instance_path(run_root, cfg.llm.sample_unit)
    if not instance_path.exists():
        materialize_llm_instances(cfg, run_root=run_root)
    instances = pd.read_parquet(instance_path)
    if instances.empty:
        write_json(llm_root / "summary.json", {"records": []})
        return

    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static = load_static_table(train_dataset.static) if cfg.llm.prompt.include_static else None
    dynamic_by_patient = _group_dynamic(dynamic)
    static_by_patient = _group_static(static)

    client = OpenAICompatibleChatClient()
    schema_text = schema_prompt_text(
        task_kind=cfg.task.kind,
        include_explanation=cfg.llm.output.include_explanation,
        include_confidence=cfg.llm.output.include_confidence,
    )

    records: list[dict[str, object]] = []
    for llm_model in cfg.llm_models:
        model_name = llm_model.name
        frame = _run_model_predictions(
            cfg=cfg,
            model_name=model_name,
            provider_model=llm_model.model,
            client=client,
            instances=instances,
            dynamic_by_patient=dynamic_by_patient,
            static_by_patient=static_by_patient,
            schema_text=schema_text,
            base_url=llm_model.base_url,
            api_key_env=llm_model.api_key_env,
            system_prompt=llm_model.system_prompt,
            supports_json_schema=llm_model.supports_json_schema,
            headers=llm_model.headers,
        )

        for split_name, split_df in frame.groupby("split", sort=True):
            split_df = split_df.sort_values(["patient_id", "instance_id"], kind="stable").reset_index(drop=True)
            prompt_rows = split_df[["instance_id", "patient_id", "split", "split_role", "prompt", "prompt_sha256"]].to_dict(orient="records")
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
            if cfg.llm.save_prompts:
                prompt_path = ensure_dir(llm_root / "prompts" / model_name) / f"{split_name}.jsonl"
                write_jsonl(prompt_path, prompt_rows)
            if cfg.llm.save_responses:
                response_path = ensure_dir(llm_root / "responses" / model_name) / f"{split_name}.jsonl"
                write_jsonl(response_path, response_rows)

            if cfg.llm.save_parsed:
                parsed_path = ensure_dir(llm_root / "parsed" / model_name) / f"{split_name}.parquet"
                parsed_cols = [
                    "instance_id",
                    "patient_id",
                    "split",
                    "split_role",
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

            pred_path = ensure_dir(llm_root / "preds" / model_name) / f"{split_name}.parquet"
            pred_cols = [
                "instance_id",
                "patient_id",
                "split",
                "split_role",
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

            if (split_df["parsed_ok"] == False).any():  # noqa: E712
                failures = split_df[split_df["parsed_ok"] == False].copy()  # noqa: E712
                failure_path = ensure_dir(llm_root / "failures" / model_name) / f"{split_name}.jsonl"
                write_jsonl(
                    failure_path,
                    failures[
                        [
                            "instance_id",
                            "patient_id",
                            "split",
                            "split_role",
                            "error_code",
                            "error_message",
                            "raw_response",
                        ]
                    ].to_dict(orient="records"),
                )

            metrics = summarize_prediction_rows(split_df, task_kind=cfg.task.kind)
            metrics_path = ensure_dir(llm_root / "metrics" / model_name) / f"{split_name}.json"
            write_json(metrics_path, metrics)
            records.append(
                {
                    "llm_model": model_name,
                    "provider_model": llm_model.model,
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
        llm_root / "summary.json",
        {
            "run_name": str(cfg.output.run_name),
            "task": {"kind": str(cfg.task.kind), "prediction_mode": str(cfg.task.prediction_mode)},
            "llm": {
                "sample_unit": str(cfg.llm.sample_unit),
                "prompt_template": str(cfg.llm.prompt_template),
                "json_schema_version": int(cfg.llm.json_schema_version),
            },
            "records": records,
        },
    )


def _run_model_predictions(
    *,
    cfg,
    model_name: str,
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
        include_explanation=cfg.llm.output.include_explanation,
        include_confidence=cfg.llm.output.include_confidence,
        supports_json_schema=supports_json_schema,
    )

    jobs = instances.to_dict(orient="records")
    if cfg.llm.concurrency <= 1:
        rows = [
            _predict_one(
                cfg=cfg,
                instance=job,
                model_name=model_name,
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
        with ThreadPoolExecutor(max_workers=int(cfg.llm.concurrency)) as ex:
            rows = list(
                ex.map(
                    lambda job: _predict_one(
                        cfg=cfg,
                        instance=job,
                        model_name=model_name,
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
    model_name: str,
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

    req = LLMRequestSpec(
        model_name=model_name,
        provider_model=provider_model,
        base_url=base_url,
        api_key_env=api_key_env,
        prompt=prompt,
        system_prompt=system_prompt,
        response_format=response_format,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        timeout_seconds=cfg.llm.timeout_seconds,
        max_retries=cfg.llm.max_retries,
        seed=cfg.llm.seed,
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
        resp = client.complete(req)
        raw_text = resp.raw_text
        latency_ms = resp.latency_ms
        usage_prompt = resp.usage_prompt_tokens
        usage_completion = resp.usage_completion_tokens
        usage_total = resp.usage_total_tokens
        parsed = parse_prediction_response(
            raw_text,
            task_kind=cfg.task.kind,
            include_explanation=cfg.llm.output.include_explanation,
            include_confidence=cfg.llm.output.include_confidence,
        )
    except LLMClientError as exc:
        error_code = exc.code
        error_message = exc.message
        raw_text = exc.response_text or ""
        parsed = parse_prediction_response(
            raw_text,
            task_kind=cfg.task.kind,
            include_explanation=cfg.llm.output.include_explanation,
            include_confidence=cfg.llm.output.include_confidence,
        ) if raw_text else None

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
        "llm_model": model_name,
        "provider_model": provider_model,
    }
    if "bin_time" in instance:
        row["bin_time"] = pd.to_datetime(instance["bin_time"], errors="raise")
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
