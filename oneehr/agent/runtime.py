from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from oneehr.agent.client import AgentClientError
from oneehr.agent.contracts import AgentRequestSpec

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
