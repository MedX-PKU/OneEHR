from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from oneehr.agent.client import OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec
from oneehr.agent.predict_schema import parse_prediction_response
from oneehr.agent.templates import describe_prompt_template, get_prompt_template, list_prompt_templates
from oneehr.config.load import load_experiment_config
from oneehr.config.schema import (
    DatasetConfig,
    DynamicTableConfig,
    ExperimentConfig,
    ModelConfig,
    SplitConfig,
    TaskConfig,
)


def test_load_config_rejects_legacy_agent_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[dataset]",
                'dynamic = "dynamic.csv"',
                "",
                "[task]",
                'kind = "binary"',
                'prediction_mode = "patient"',
                "",
                "[split]",
                'kind = "random"',
                "",
                "[agent.predict]",
                "enabled = true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy config sections are no longer supported"):
        load_experiment_config(cfg_path)


def test_prompt_template_registry() -> None:
    templates = list_prompt_templates()
    names = {item["name"] for item in templates}
    assert names == {"summary_v1"}

    summary = describe_prompt_template("summary_v1")
    assert summary["family"] == "prediction"
    assert "prediction_task" in summary["default_sections"]


def test_parse_prediction_response_binary_and_regression() -> None:
    parsed = parse_prediction_response(
        '{"label": 1, "explanation": "high risk"}',
        task_kind="binary",
        include_explanation=True,
        include_confidence=False,
    )
    assert parsed.parsed_ok is True
    assert parsed.prediction == 1
    assert parsed.probability == 1.0
    assert parsed.explanation == "high risk"

    parsed_reg = parse_prediction_response(
        '```json\n{"value": 3.5}\n```',
        task_kind="regression",
        include_explanation=False,
        include_confidence=False,
    )
    assert parsed_reg.parsed_ok is True
    assert parsed_reg.value == 3.5

    bad = parse_prediction_response(
        "not json",
        task_kind="binary",
        include_explanation=False,
        include_confidence=False,
    )
    assert bad.parsed_ok is False
    assert bad.error_code == "invalid_json"


def test_render_prompt_time_excludes_future_events() -> None:
    cfg = ExperimentConfig(
        dataset=DatasetConfig(dynamic=DynamicTableConfig(path=Path("dynamic.csv"))),
        task=TaskConfig(kind="binary", prediction_mode="time"),
        split=SplitConfig(kind="random"),
        model=ModelConfig(name="xgboost"),
    )
    dynamic = pd.DataFrame(
        [
            {"patient_id": "p1", "event_time": "2020-01-01", "code": "LAB_PAST", "value": 1.0},
            {"patient_id": "p1", "event_time": "2020-01-03", "code": "LAB_FUTURE", "value": 9.0},
        ]
    )
    dynamic["event_time"] = pd.to_datetime(dynamic["event_time"])

    spec = get_prompt_template("summary_v1")
    assert spec.renderer is not None
    prompt = spec.renderer(
        cfg=cfg,
        instance={
            "instance_id": "time0:p1:2020-01-02T00:00:00",
            "patient_id": "p1",
            "split": "time0",
            "split_role": "test",
            "bin_time": pd.Timestamp("2020-01-02"),
        },
        dynamic=dynamic,
        static_row=None,
        schema_text="{}",
    )

    assert "LAB_PAST" in prompt
    assert "LAB_FUTURE" not in prompt


class _MockChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(body)  # type: ignore[attr-defined]

        status_plan = self.server.status_plan  # type: ignore[attr-defined]
        status = status_plan.pop(0) if status_plan else 200
        if status != 200:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"temporary"}}')
            return

        prompt = str(body["messages"][-1]["content"])
        match = re.search(r"patient_id: (p\d+)", prompt)
        patient_id = match.group(1) if match else "p0000"
        label = int(patient_id[1:]) % 2
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "label": label,
                                "probability": float(label),
                                "explanation": f"derived from {patient_id}",
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


@contextmanager
def _mock_chat_server(*, status_plan: list[int] | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockChatHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.status_plan = list(status_plan or [])  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def test_openai_compatible_agent_client_retries_on_500(monkeypatch) -> None:
    with _mock_chat_server(status_plan=[500]) as (_, base_url):
        monkeypatch.setenv("TEST_OPENAI_API_KEY", "dummy")
        client = OpenAICompatibleAgentClient()
        resp = client.complete(
            AgentRequestSpec(
                backend_name="mock",
                provider_model="mock-model",
                base_url=base_url,
                api_key_env="TEST_OPENAI_API_KEY",
                prompt="patient_id: p0001",
                system_prompt=None,
                response_format={"type": "json_object"},
                temperature=0.0,
                top_p=1.0,
                timeout_seconds=5.0,
                max_retries=1,
                seed=None,
            )
        )
        assert '"label": 1' in resp.raw_text
