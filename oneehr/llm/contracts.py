from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMRequestSpec:
    model_name: str
    provider_model: str
    base_url: str
    api_key_env: str
    prompt: str
    system_prompt: str | None
    response_format: dict[str, Any] | None
    temperature: float
    top_p: float
    timeout_seconds: float
    max_retries: int
    seed: int | None
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    raw_text: str
    response_json: dict[str, Any]
    latency_ms: int
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    usage_total_tokens: int | None = None


@dataclass(frozen=True)
class ParsedPrediction:
    parsed_ok: bool
    prediction: float | int | None
    probability: float | None = None
    value: float | None = None
    explanation: str | None = None
    confidence: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    raw_json: dict[str, Any] | None = None

