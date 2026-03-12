from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from oneehr.agent.client import OpenAICompatibleAgentClient
from oneehr.agent.contracts import AgentRequestSpec
from oneehr.agent.instances import validate_agent_predict_setup
from oneehr.agent.predict_schema import parse_prediction_response
from oneehr.agent.render import render_prompt
from oneehr.agent.templates import describe_prompt_template, list_prompt_templates
from oneehr.config.load import load_experiment_config
from oneehr.config.schema import (
    AgentBackendConfig,
    AgentConfig,
    AgentPredictConfig,
    AgentPredictOutputConfig,
    AgentPredictPromptConfig,
    DatasetConfig,
    DynamicTableConfig,
    ExperimentConfig,
    ModelConfig,
    SplitConfig,
    TaskConfig,
)


def test_load_config_with_agent_predict_sections(tmp_path: Path) -> None:
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
                'sample_unit = "patient"',
                'prompt_template = "summary_v1"',
                "concurrency = 2",
                "",
                "[agent.predict.prompt]",
                "max_events = 25",
                'time_order = "desc"',
                "",
                "[agent.predict.output]",
                "include_explanation = true",
                "",
                "[[agent.predict.backends]]",
                'name = "mock"',
                'provider = "openai_compatible"',
                'base_url = "http://127.0.0.1:9999/v1"',
                'model = "mock-model"',
                'api_key_env = "TEST_OPENAI_API_KEY"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(cfg_path)
    assert cfg.agent.predict.enabled is True
    assert cfg.agent.predict.concurrency == 2
    assert cfg.agent.predict.prompt.max_events == 25
    assert cfg.agent.predict.prompt.time_order == "desc"
    assert len(cfg.agent.predict.backends) == 1
    assert cfg.model is None
    assert cfg.agent.predict.backends[0].base_url == "http://127.0.0.1:9999/v1"
    validate_agent_predict_setup(cfg)
    with pytest.raises(ValueError, match="training"):
        cfg.require_model(context="training")


def test_prompt_template_registry() -> None:
    templates = list_prompt_templates()
    names = {item["name"] for item in templates}
    assert {"summary_v1", "evidence_review_v1"} <= names

    summary = describe_prompt_template("summary_v1")
    assert summary["family"] == "prediction"
    assert "prediction_task" in summary["default_sections"]

    review = describe_prompt_template("evidence_review_v1")
    assert review["family"] == "review"
    assert review["allow_labels_context"] is True


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
        agent=AgentConfig(
            predict=AgentPredictConfig(
                enabled=True,
                sample_unit="time",
                prompt=AgentPredictPromptConfig(max_events=10),
                output=AgentPredictOutputConfig(include_explanation=True),
                backends=[AgentBackendConfig(name="mock", model="mock-model")],
            )
        ),
    )
    dynamic = pd.DataFrame(
        [
            {"patient_id": "p1", "event_time": "2020-01-01", "code": "LAB_PAST", "value": 1.0},
            {"patient_id": "p1", "event_time": "2020-01-03", "code": "LAB_FUTURE", "value": 9.0},
        ]
    )
    dynamic["event_time"] = pd.to_datetime(dynamic["event_time"])

    prompt = render_prompt(
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


def test_agent_predict_cli_e2e_patient_binary(tmp_path: Path) -> None:
    dynamic_csv = tmp_path / "dynamic.csv"
    label_fn = tmp_path / "label_fn.py"
    cfg_path = tmp_path / "exp.toml"
    out_root = tmp_path / "runs"
    run_name = "agent_patient"

    _write_dynamic_csv(dynamic_csv)
    _write_patient_parity_label_fn(label_fn)

    with _mock_chat_server() as (server, base_url):
        _write_agent_predict_experiment_toml(
            path=cfg_path,
            dynamic_csv=dynamic_csv,
            label_fn_ref=f"{label_fn}:build_labels",
            out_root=out_root,
            run_name=run_name,
            base_url=base_url,
        )
        env = os.environ.copy()
        env["TEST_OPENAI_API_KEY"] = "dummy"

        subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg_path)], env=env)
        subprocess.check_call(["oneehr", "agent", "predict", "--config", str(cfg_path)], env=env)

        run_root = out_root / run_name
        instances_path = run_root / "agent" / "predict" / "instances" / "patient_instances.parquet"
        summary_path = run_root / "agent" / "predict" / "summary.json"
        preds_path = run_root / "agent" / "predict" / "preds" / "mock" / "split0.parquet"

        assert instances_path.exists()
        assert preds_path.exists()
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert len(summary["records"]) == 1
        record = summary["records"][0]
        assert record["predictor_name"] == "mock"
        assert record["parse_success_rate"] == 1.0
        assert record["coverage"] > 0.0
        assert record["metrics"]["accuracy"] == 1.0
        assert len(server.requests) > 0


def _write_dynamic_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for pid_int in range(12):
        patient_id = f"p{pid_int:04d}"
        rows.append({"patient_id": patient_id, "event_time": "2020-01-01", "code": "LAB_A", "value": pid_int})
        rows.append({"patient_id": patient_id, "event_time": "2020-01-02", "code": "MED_X", "value": pid_int % 2})
        rows.append({"patient_id": patient_id, "event_time": "2020-01-03", "code": "LAB_B", "value": pid_int + 0.5})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_patient_parity_label_fn(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import pandas as pd",
                "",
                "",
                "def build_labels(dynamic: pd.DataFrame, static, label, cfg):",
                "    pids = dynamic[['patient_id']].drop_duplicates().copy()",
                "    pids['patient_id'] = pids['patient_id'].astype(str)",
                "    pids['label'] = pids['patient_id'].str.extract(r'(\\d+)', expand=False).astype(int) % 2",
                "    return pids[['patient_id', 'label']]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_agent_predict_experiment_toml(
    *,
    path: Path,
    dynamic_csv: Path,
    label_fn_ref: str,
    out_root: Path,
    run_name: str,
    base_url: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "[dataset]",
                f'dynamic = "{dynamic_csv}"',
                "",
                "[preprocess]",
                'bin_size = "1d"',
                'numeric_strategy = "mean"',
                'categorical_strategy = "onehot"',
                'code_selection = "frequency"',
                "top_k_codes = 20",
                "",
                "[task]",
                'kind = "binary"',
                'prediction_mode = "patient"',
                "",
                "[labels]",
                f'fn = "{label_fn_ref}"',
                "",
                "[split]",
                'kind = "random"',
                "seed = 0",
                "val_size = 0.2",
                "test_size = 0.25",
                "",
                "[trainer]",
                'device = "cpu"',
                "repeat = 1",
                "",
                "[agent.predict]",
                "enabled = true",
                'sample_unit = "patient"',
                'prompt_template = "summary_v1"',
                "json_schema_version = 1",
                "concurrency = 1",
                "max_retries = 1",
                "timeout_seconds = 5.0",
                "temperature = 0.0",
                "top_p = 1.0",
                "",
                "[agent.predict.prompt]",
                "include_static = false",
                "max_events = 20",
                'time_order = "asc"',
                'sections = ["patient_profile", "event_timeline", "prediction_task", "output_schema"]',
                "",
                "[agent.predict.output]",
                "include_explanation = true",
                "include_confidence = false",
                "",
                "[[agent.predict.backends]]",
                'name = "mock"',
                'provider = "openai_compatible"',
                f'base_url = "{base_url}"',
                'model = "mock-model"',
                'api_key_env = "TEST_OPENAI_API_KEY"',
                "supports_json_schema = true",
                "",
                "[output]",
                f'root = "{out_root}"',
                f'run_name = "{run_name}"',
                "save_preds = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
