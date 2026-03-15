from __future__ import annotations

from hashlib import sha256

from oneehr.agent.client import AgentClientError
from oneehr.agent.contracts import AgentRequestSpec, AgentResponse
from oneehr.agent.predict_schema import parse_prediction_response
from oneehr.agent.runtime import execute_agent_request, run_jobs


class _StaticClient:
    def __init__(self, response: AgentResponse) -> None:
        self._response = response

    def complete(self, spec: AgentRequestSpec) -> AgentResponse:
        return self._response


class _FailingClient:
    def __init__(self, error: AgentClientError) -> None:
        self._error = error

    def complete(self, spec: AgentRequestSpec) -> AgentResponse:
        raise self._error


def test_execute_agent_request_tracks_hashes_usage_and_parse_status() -> None:
    request = AgentRequestSpec(
        backend_name="mock",
        provider_model="mock-model",
        base_url="http://127.0.0.1:9999/v1",
        api_key_env="TEST_OPENAI_API_KEY",
        prompt="patient_id: p0001",
        system_prompt=None,
        response_format={"type": "json_object"},
        temperature=0.0,
        top_p=1.0,
        timeout_seconds=5.0,
        max_retries=0,
        seed=7,
    )
    response = AgentResponse(
        raw_text='{"label": 1, "probability": 0.9}',
        response_json={},
        latency_ms=42,
        usage_prompt_tokens=11,
        usage_completion_tokens=6,
        usage_total_tokens=17,
    )

    result = execute_agent_request(
        client=_StaticClient(response),
        request=request,
        parse_response=lambda raw_text: parse_prediction_response(
            raw_text,
            task_kind="binary",
            include_explanation=False,
            include_confidence=False,
        ),
    )

    assert result.parsed is not None
    assert result.parsed.parsed_ok is True
    assert result.prompt_sha256 == sha256(request.prompt.encode("utf-8")).hexdigest()
    assert result.response_sha256 == sha256(response.raw_text.encode("utf-8")).hexdigest()
    assert result.latency_ms == 42
    assert result.token_usage_total == 17
    assert result.error_code is None


def test_execute_agent_request_normalizes_parse_errors() -> None:
    request = AgentRequestSpec(
        backend_name="mock",
        provider_model="mock-model",
        base_url="http://127.0.0.1:9999/v1",
        api_key_env="TEST_OPENAI_API_KEY",
        prompt="patient_id: p0002",
        system_prompt=None,
        response_format={"type": "json_object"},
        temperature=0.0,
        top_p=1.0,
        timeout_seconds=5.0,
        max_retries=0,
        seed=None,
    )

    result = execute_agent_request(
        client=_StaticClient(AgentResponse(raw_text="not json", response_json={}, latency_ms=10)),
        request=request,
        parse_response=lambda raw_text: parse_prediction_response(
            raw_text,
            task_kind="binary",
            include_explanation=False,
            include_confidence=False,
        ),
    )

    assert result.parsed is not None
    assert result.parsed.parsed_ok is False
    assert result.error_code == "invalid_json"
    assert result.error_message == "response did not contain a JSON object"


def test_execute_agent_request_reuses_error_response_body_for_parse() -> None:
    request = AgentRequestSpec(
        backend_name="mock-predict",
        provider_model="mock-predict-model",
        base_url="http://127.0.0.1:9999/v1",
        api_key_env="TEST_OPENAI_API_KEY",
        prompt="predict case",
        system_prompt=None,
        response_format={"type": "json_object"},
        temperature=0.0,
        top_p=1.0,
        timeout_seconds=5.0,
        max_retries=0,
        seed=None,
    )
    result = execute_agent_request(
        client=_FailingClient(
            AgentClientError(
                code="http_error",
                message="provider returned HTTP 500",
                status_code=500,
                response_text=(
                    '{"label": 1, "probability": 0.8, "explanation": "supported by evidence"}'
                ),
            )
        ),
        request=request,
        parse_response=lambda raw_text: parse_prediction_response(
            raw_text,
            task_kind="binary",
            include_explanation=True,
            include_confidence=False,
        ),
    )

    assert result.parsed is not None
    assert result.parsed.parsed_ok is True
    assert result.error_code == "http_error"
    assert result.response_sha256 is not None


def test_run_jobs_preserves_input_order() -> None:
    assert run_jobs([1, 2, 3], worker=lambda value: value * 2, concurrency=1) == [2, 4, 6]
    assert run_jobs([1, 2, 3], worker=lambda value: value * 3, concurrency=2) == [3, 6, 9]
