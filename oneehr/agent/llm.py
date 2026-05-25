"""Shared LLM request helpers for OneEHR agent frameworks."""

from __future__ import annotations

from typing import Callable, TypeVar

from oneehr.agent.client import OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec, ParsedPrediction
from oneehr.agent.runtime import AgentExecutionResult, execute_agent_request
from oneehr.agent.schema import parse_prediction_response, schema_prompt_text
from oneehr.config.schema import SystemConfig

ParsedT = TypeVar("ParsedT")


def build_llm_request(
    system_cfg: SystemConfig,
    prompt: str,
    system_prompt: str | None = None,
) -> AgentRequestSpec:
    params = system_cfg.params or {}
    backend = system_cfg.backend.lower()
    base_urls = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "together": "https://api.together.xyz/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    base_url = str(params.get("base_url", base_urls.get(backend, "https://api.openai.com/v1")))

    return AgentRequestSpec(
        backend_name=backend,
        provider_model=system_cfg.model,
        base_url=base_url,
        api_key_env=system_cfg.api_key_env,
        prompt=prompt,
        system_prompt=system_prompt,
        response_format=None,
        temperature=float(params.get("temperature", 0.3)),
        top_p=float(params.get("top_p", 1.0)),
        timeout_seconds=float(params.get("timeout_seconds", 120.0)),
        max_retries=int(params.get("max_retries", 2)),
        seed=int(params["seed"]) if "seed" in params else None,
    )


def call_llm_text(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    prompt: str,
    *,
    system_prompt: str | None = None,
    parse_response: Callable[[str], ParsedT | None],
) -> AgentExecutionResult[ParsedT]:
    request = build_llm_request(system_cfg, prompt, system_prompt)
    return execute_agent_request(
        client=client,
        request=request,
        parse_response=parse_response,
    )


def call_prediction_llm(
    client: OpenAICompatibleAgentClient,
    system_cfg: SystemConfig,
    prompt: str,
    task_kind: str,
    *,
    system_prompt: str | None = None,
) -> ParsedPrediction | None:
    result = call_llm_text(
        client,
        system_cfg,
        prompt,
        system_prompt=system_prompt,
        parse_response=lambda text: parse_prediction_response(
            text,
            task_kind=task_kind,
            include_explanation=True,
            include_confidence=True,
        ),
    )
    return result.parsed


def prediction_schema_text(task_kind: str) -> str:
    return schema_prompt_text(
        task_kind=task_kind,
        include_explanation=True,
        include_confidence=True,
    )
