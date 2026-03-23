from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import pandas as pd

from oneehr.agent.client import AgentClientError, OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec
from oneehr.config.schema import SystemConfig

ParsedT = TypeVar("ParsedT")
JobT = TypeVar("JobT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class AgentExecutionResult(Generic[ParsedT]):
    parsed: ParsedT | None
    raw_response: str
    prompt_sha256: str
    response_sha256: str | None
    latency_ms: int | None
    token_usage_prompt: int | None
    token_usage_completion: int | None
    token_usage_total: int | None
    error_code: str | None
    error_message: str | None


def run_jobs(
    jobs: list[JobT],
    *,
    worker: Callable[[JobT], ResultT],
    concurrency: int,
) -> list[ResultT]:
    if int(concurrency) <= 1:
        return [worker(job) for job in jobs]

    with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
        return list(executor.map(worker, jobs))


def execute_agent_request(
    *,
    client: Any,
    request: AgentRequestSpec,
    parse_response: Callable[[str], ParsedT | None],
) -> AgentExecutionResult[ParsedT]:
    prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()

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
        parsed = parse_response(raw_text)
    except AgentClientError as exc:
        error_code = exc.code
        error_message = exc.message
        raw_text = exc.response_text or ""
        parsed = parse_response(raw_text) if raw_text else None

    if parsed is None and error_code is None:
        error_code = "request_failed"
        error_message = "request failed before a response was returned"
    elif parsed is not None and not getattr(parsed, "parsed_ok", False) and error_code is None:
        error_code = getattr(parsed, "error_code", None)
        error_message = getattr(parsed, "error_message", None)

    response_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else None
    return AgentExecutionResult(
        parsed=parsed,
        raw_response=raw_text,
        prompt_sha256=prompt_sha256,
        response_sha256=response_sha256,
        latency_ms=latency_ms,
        token_usage_prompt=usage_prompt,
        token_usage_completion=usage_completion,
        token_usage_total=usage_total,
        error_code=error_code,
        error_message=error_message,
    )


def _build_patient_context(
    patient_id: str,
    binned: pd.DataFrame,
    feat_cols: list[str],
    task_kind: str,
) -> str:
    """Build a text summary of a patient's clinical data for LLM consumption."""
    pdf = binned[binned["patient_id"].astype(str) == patient_id].copy()
    if pdf.empty:
        return f"Patient ID: {patient_id}\nNo clinical data available."

    pdf = pdf.sort_values("bin_time", kind="stable")
    lines = [f"Patient ID: {patient_id}", ""]

    # Latest values
    last_row = pdf[feat_cols].iloc[-1]
    lines.append("Latest clinical values:")
    for col in feat_cols:
        val = last_row[col]
        if pd.notna(val):
            clean_name = col.replace("num__", "").replace("cat__", "")
            lines.append(f"  {clean_name}: {val}")

    # Time series summary
    n_visits = len(pdf)
    lines.append(f"\nNumber of time points: {n_visits}")
    if "bin_time" in pdf.columns:
        lines.append(f"Time range: {pdf['bin_time'].iloc[0]} to {pdf['bin_time'].iloc[-1]}")

    return "\n".join(lines)


def run_system_on_patients(
    *,
    system_cfg: SystemConfig,
    binned: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    feat_cols: list[str],
    test_pids: set[str],
    task_kind: str,
) -> list[dict]:
    """Run an LLM agent framework on test patients and return prediction rows.

    Returns list of dicts with keys: system, patient_id, y_true, y_pred.
    """
    from oneehr.agent.frameworks import get_framework

    framework_runner = get_framework(system_cfg.framework)
    client = OpenAICompatibleAgentClient()

    # Build y_true map
    y_true_map: dict[str, float] = {}
    if labels_df is not None:
        for _, row in labels_df.iterrows():
            pid = str(row["patient_id"])
            if pid in test_pids:
                y_true_map[pid] = float(row["label"])

    params = system_cfg.params or {}
    concurrency = int(params.get("concurrency", 1))

    def _predict_one(pid: str) -> dict:
        context = _build_patient_context(pid, binned, feat_cols, task_kind)
        try:
            pred = framework_runner(client, system_cfg, context, task_kind)
        except Exception as exc:
            print(f"  Warning: framework error for {pid}: {exc}")
            pred = None

        if pred is not None and pred.parsed_ok:
            if task_kind == "binary":
                y_pred = float(pred.probability if pred.probability is not None else (pred.prediction or 0))
            else:
                y_pred = float(pred.value if pred.value is not None else (pred.prediction or 0))
        else:
            y_pred = float("nan")

        return {
            "system": system_cfg.name,
            "patient_id": pid,
            "y_true": y_true_map.get(pid, float("nan")),
            "y_pred": y_pred,
        }

    pid_list = sorted(test_pids)
    return run_jobs(pid_list, worker=_predict_one, concurrency=concurrency)
