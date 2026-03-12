from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from oneehr.agent.contracts import AgentRequestSpec, AgentResponse


@dataclass(frozen=True)
class AgentClientError(RuntimeError):
    code: str
    message: str
    status_code: int | None = None
    response_text: str | None = None
    retryable: bool = False

    def __str__(self) -> str:
        suffix = f" status={self.status_code}" if self.status_code is not None else ""
        return f"{self.code}{suffix}: {self.message}"


class OpenAICompatibleAgentClient:
    def complete(self, spec: AgentRequestSpec) -> AgentResponse:
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise AgentClientError(
                code="missing_api_key",
                message=f"Environment variable {spec.api_key_env!r} is not set",
                retryable=False,
            )

        payload: dict[str, Any] = {
            "model": spec.provider_model,
            "messages": self._build_messages(spec),
            "temperature": float(spec.temperature),
            "top_p": float(spec.top_p),
        }
        if spec.response_format is not None:
            payload["response_format"] = spec.response_format
        if spec.seed is not None:
            payload["seed"] = int(spec.seed)

        body = json.dumps(payload).encode("utf-8")
        url = spec.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(spec.extra_headers)

        last_error: AgentClientError | None = None
        for attempt in range(spec.max_retries + 1):
            started = time.monotonic()
            req = request.Request(url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=spec.timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise AgentClientError(
                            code="invalid_response",
                            message="provider response must be a JSON object",
                            response_text=raw,
                            retryable=False,
                        )
                    text = _extract_message_content(payload)
                    latency_ms = int(round((time.monotonic() - started) * 1000.0))
                    usage = payload.get("usage") or {}
                    return AgentResponse(
                        raw_text=text,
                        response_json=payload,
                        latency_ms=latency_ms,
                        usage_prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                        usage_completion_tokens=_optional_int(usage.get("completion_tokens")),
                        usage_total_tokens=_optional_int(usage.get("total_tokens")),
                    )
            except error.HTTPError as exc:
                resp_text = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                last_error = AgentClientError(
                    code="http_error",
                    message=f"provider returned HTTP {exc.code}",
                    status_code=exc.code,
                    response_text=resp_text,
                    retryable=retryable,
                )
            except error.URLError as exc:
                last_error = AgentClientError(
                    code="network_error",
                    message=str(exc.reason),
                    retryable=True,
                )
            except TimeoutError as exc:
                last_error = AgentClientError(
                    code="timeout",
                    message=str(exc),
                    retryable=True,
                )

            if last_error is None:
                break
            if not last_error.retryable or attempt >= spec.max_retries:
                raise last_error
            time.sleep(min(2.0 ** attempt, 8.0) * 0.5)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _build_messages(spec: AgentRequestSpec) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if spec.system_prompt:
            messages.append({"role": "system", "content": spec.system_prompt})
        messages.append({"role": "user", "content": spec.prompt})
        return messages


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgentClientError(
            code="invalid_response",
            message="provider response missing choices[0]",
            response_text=json.dumps(payload, sort_keys=True),
            retryable=False,
        )
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        if parts:
            return "\n".join(parts)
    raise AgentClientError(
        code="invalid_response",
        message="provider response missing message.content text",
        response_text=json.dumps(payload, sort_keys=True),
        retryable=False,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
