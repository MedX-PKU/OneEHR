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

from oneehr.eval.query import read_instance_payload, read_trace_rows
from oneehr.web.service import WebUIService
from test_analysis import _build_trained_run


def test_eval_cli_e2e_all_supported_frameworks(tmp_path: Path) -> None:
    with _mock_eval_server() as (server, base_url):
        run_root, cfg_path = _build_eval_run(
            tmp_path=tmp_path,
            run_name="eval_e2e",
            seed=23,
            base_url=base_url,
        )
        env = _eval_env()

    expected_systems = {
        "xgboost_ref",
        "single_llm_eval",
        "healthcareagent_eval",
        "reconcile_eval",
        "mac_eval",
        "medagent_eval",
        "colacare_eval",
        "mdagents_eval",
    }
    eval_root = run_root / "eval"
    assert (eval_root / "index.json").exists()
    assert (eval_root / "summary.json").exists()
    assert (eval_root / "reports" / "summary.json").exists()

    index = json.loads((eval_root / "index.json").read_text(encoding="utf-8"))
    assert index["instance_count"] == 4
    assert len(index["records"]) == 4

    run_summary = json.loads((eval_root / "summary.json").read_text(encoding="utf-8"))
    assert {record["system_name"] for record in run_summary["records"]} == expected_systems

    for system_name in expected_systems:
        pred_path = eval_root / "predictions" / system_name / "predictions.parquet"
        assert pred_path.exists()
        pred_df = pd.read_parquet(pred_path)
        assert len(pred_df) == 4
        assert set(pred_df["system_name"].astype(str)) == {system_name}

    for system_name in expected_systems - {"xgboost_ref"}:
        trace_path = eval_root / "traces" / system_name / "trace.parquet"
        assert trace_path.exists()
        trace_df = pd.read_parquet(trace_path)
        assert not trace_df.empty
        assert set(trace_df["system_name"].astype(str)) == {system_name}

    leaderboard = pd.read_csv(eval_root / "reports" / "leaderboard.csv")
    assert set(leaderboard["system_name"].astype(str)) == expected_systems
    assert "accuracy" in leaderboard.columns

    pairwise = pd.read_csv(eval_root / "reports" / "pairwise.csv")
    assert not pairwise.empty
    assert set(pairwise["suite_name"].astype(str)) == {"core"}

    first_instance_id = str(index["records"][0]["instance_id"])
    payload = read_instance_payload(run_root, instance_id=first_instance_id)
    assert payload["instance_id"] == first_instance_id
    assert len(payload["outputs"]) == len(expected_systems)
    assert len(payload["evidence"]["events"]) >= 1

    trace_payload = read_trace_rows(
        run_root,
        system_name="mdagents_eval",
        limit=10,
        stage="complexity_routing",
    )
    assert trace_payload["total_rows"] >= 1
    assert trace_payload["row_count"] >= 1
    assert all(record["stage"] == "complexity_routing" for record in trace_payload["records"])

    cli_instance = json.loads(
        subprocess.check_output(
            [
                "oneehr",
                "eval",
                "instance",
                "--config",
                str(cfg_path),
                "--instance-id",
                first_instance_id,
            ],
            env=env,
            text=True,
        )
    )
    assert cli_instance["instance_id"] == first_instance_id
    assert len(cli_instance["outputs"]) == len(expected_systems)

    cli_trace = json.loads(
        subprocess.check_output(
            [
                "oneehr",
                "eval",
                "trace",
                "--config",
                str(cfg_path),
                "--system",
                "healthcareagent_eval",
                "--stage",
                "plan",
                "--limit",
                "5",
            ],
            env=env,
            text=True,
        )
    )
    assert cli_trace["system_name"] == "healthcareagent_eval"
    assert cli_trace["row_count"] >= 1

    query_report = json.loads(
        subprocess.check_output(
            ["oneehr", "query", "eval", "report", "--config", str(cfg_path)],
            env=env,
            text=True,
        )
    )
    assert query_report["report"]["primary_metric"] == "accuracy"

    query_table = json.loads(
        subprocess.check_output(
            [
                "oneehr",
                "query",
                "eval",
                "table",
                "--config",
                str(cfg_path),
                "--table",
                "leaderboard",
                "--limit",
                "2",
            ],
            env=env,
            text=True,
        )
    )
    assert query_table["table"]["table"] == "leaderboard"
    assert query_table["table"]["row_count"] == 2

    service = WebUIService(root_dir=run_root.parent)
    eval_payload = service.eval_payload(run_name=run_root.name)
    assert eval_payload["status"] == "ok"
    assert eval_payload["index"]["instance_count"] == 4
    assert any(table["name"] == "leaderboard" for table in eval_payload["tables"])

    eval_table_payload = service.eval_table_payload(
        run_name=run_root.name,
        table_name="leaderboard",
        limit=3,
        offset=0,
        sort_by="accuracy",
        sort_dir="desc",
    )
    assert eval_table_payload["table"] == "leaderboard"
    assert eval_table_payload["row_count"] == 3

    eval_instance_payload = service.eval_instance_payload(
        run_name=run_root.name,
        instance_id=first_instance_id,
    )
    assert eval_instance_payload["instance"]["instance_id"] == first_instance_id

    eval_trace_payload = service.eval_trace_payload(
        run_name=run_root.name,
        system_name="healthcareagent_eval",
        stage="plan",
        limit=5,
    )
    assert eval_trace_payload["row_count"] >= 1
    assert eval_trace_payload["table"] == "trace_rows"

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from oneehr.web import create_app

    client = TestClient(create_app(root_dir=run_root.parent))
    eval_route = client.get(f"/api/v1/runs/{run_root.name}/eval")
    assert eval_route.status_code == 200
    assert eval_route.json()["status"] == "ok"

    table_route = client.get(
        f"/api/v1/runs/{run_root.name}/eval/tables/leaderboard",
        params={"limit": 2, "sort_by": "accuracy", "sort_dir": "desc"},
    )
    assert table_route.status_code == 200
    assert table_route.json()["table"] == "leaderboard"
    assert table_route.json()["row_count"] == 2

    instance_route = client.get(f"/api/v1/runs/{run_root.name}/eval/instances/{first_instance_id}")
    assert instance_route.status_code == 200
    assert instance_route.json()["instance"]["instance_id"] == first_instance_id

    trace_route = client.get(
        f"/api/v1/runs/{run_root.name}/eval/traces/healthcareagent_eval",
        params={"stage": "plan", "limit": 5},
    )
    assert trace_route.status_code == 200
    assert trace_route.json()["row_count"] >= 1
    assert len(server.requests) > 0


def _build_eval_run(*, tmp_path: Path, run_name: str, seed: int, base_url: str) -> tuple[Path, Path]:
    run_root, cfg_path = _build_trained_run(
        tmp_path=tmp_path,
        run_name=run_name,
        seed=seed,
    )
    _append_eval_config(cfg_path, base_url=base_url)
    env = _eval_env()
    subprocess.check_call(["oneehr", "eval", "build", "--config", str(cfg_path), "--force"], env=env)
    subprocess.check_call(["oneehr", "eval", "run", "--config", str(cfg_path), "--force"], env=env)
    subprocess.check_call(["oneehr", "eval", "report", "--config", str(cfg_path), "--force"], env=env)
    return run_root, cfg_path


def _eval_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TEST_OPENAI_API_KEY"] = "dummy"
    return env


def _append_eval_config(path: Path, *, base_url: str) -> None:
    eval_block = "\n".join(
        [
            "",
            "[eval]",
            'instance_unit = "patient"',
            "max_instances = 4",
            "seed = 13",
            "include_static = false",
            "include_analysis_context = false",
            "max_events = 6",
            'time_order = "asc"',
            'primary_metric = "accuracy"',
            "bootstrap_samples = 8",
            "save_evidence = true",
            "save_traces = true",
            "",
            "[[eval.backends]]",
            'name = "mock_eval"',
            'provider = "openai_compatible"',
            f'base_url = "{base_url}"',
            'model = "mock-eval-model"',
            'api_key_env = "TEST_OPENAI_API_KEY"',
            "supports_json_schema = true",
            "prompt_token_cost_per_1k = 0.002",
            "completion_token_cost_per_1k = 0.004",
            "",
            "[[eval.systems]]",
            'name = "xgboost_ref"',
            'kind = "trained_model"',
            'sample_unit = "patient"',
            'source_model = "xgboost"',
            "",
            "[[eval.systems]]",
            'name = "single_llm_eval"',
            'kind = "framework"',
            'framework_type = "single_llm"',
            'sample_unit = "patient"',
            'backend_refs = ["mock_eval"]',
            "concurrency = 2",
            "max_retries = 0",
            "timeout_seconds = 5.0",
            "",
            "[[eval.systems]]",
            'name = "healthcareagent_eval"',
            'kind = "framework"',
            'framework_type = "healthcareagent"',
            'sample_unit = "patient"',
            'backend_refs = ["mock_eval"]',
            "max_rounds = 1",
            "max_retries = 0",
            "timeout_seconds = 5.0",
            "",
            "[[eval.systems]]",
            'name = "reconcile_eval"',
            'kind = "framework"',
            'framework_type = "reconcile"',
            'sample_unit = "patient"',
            'backend_refs = ["mock_eval"]',
            "max_rounds = 2",
            "max_retries = 0",
            "timeout_seconds = 5.0",
            "",
            "[[eval.systems]]",
            'name = "mac_eval"',
            'kind = "framework"',
            'framework_type = "mac"',
            'sample_unit = "patient"',
            'backend_refs = ["mock_eval"]',
            "max_rounds = 2",
            "max_retries = 0",
            "timeout_seconds = 5.0",
            "",
            "[[eval.systems]]",
            'name = "medagent_eval"',
            'kind = "framework"',
            'framework_type = "medagent"',
            'sample_unit = "patient"',
            'backend_refs = ["mock_eval"]',
            "max_rounds = 2",
            "max_retries = 0",
            "timeout_seconds = 5.0",
            "",
            "[[eval.systems]]",
            'name = "colacare_eval"',
            'kind = "framework"',
            'framework_type = "colacare"',
            'sample_unit = "patient"',
            'backend_refs = ["mock_eval"]',
            "max_rounds = 2",
            "max_retries = 0",
            "timeout_seconds = 5.0",
            "",
            "[[eval.systems]]",
            'name = "mdagents_eval"',
            'kind = "framework"',
            'framework_type = "mdagents"',
            'sample_unit = "patient"',
            'backend_refs = ["mock_eval"]',
            "max_rounds = 2",
            "max_retries = 0",
            "timeout_seconds = 5.0",
            "",
            "[[eval.suites]]",
            'name = "core"',
            'primary_metric = "accuracy"',
            'compare_pairs = [["xgboost_ref", "single_llm_eval"], ["xgboost_ref", "mdagents_eval"]]',
            "",
        ]
    )
    path.write_text(path.read_text(encoding="utf-8") + eval_block, encoding="utf-8")


class _EvalMockHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(body)  # type: ignore[attr-defined]

        prompt = str(body["messages"][-1]["content"])
        patient_id = _extract_patient_id(prompt)
        patient_num = int(patient_id[1:]) if patient_id.startswith("p") else 0
        label = patient_num % 2
        probability = 0.85 if label == 1 else 0.15

        if "Choose the safest first step" in prompt:
            content = {"answer": "INQUIRY" if patient_num % 2 == 0 else "DIAGNOSE"}
        elif "Generate up to three critical follow-up questions" in prompt:
            content = {
                "questions": [
                    f"confirm timeline for {patient_id}",
                    "check recent medication changes",
                ]
            }
        elif "Return JSON only with {'feedback': '...'}." in prompt:
            content = {"feedback": f"mock safety feedback for {patient_id}"}
        elif "Classify the case complexity." in prompt:
            complexity = ["basic", "intermediate", "advanced"][patient_num % 3]
            content = {"complexity": complexity}
        elif "Choose the most relevant specialties for this case." in prompt:
            content = {
                "specialties": [
                    "general_medicine",
                    "cardiology",
                    "critical_care",
                ]
            }
        elif "Return JSON only with {'agree': true|false, 'reason': '...'}." in prompt:
            content = {"agree": True, "reason": f"mock review agrees for {patient_id}"}
        else:
            content = {
                "label": label,
                "probability": probability,
                "explanation": f"mock prediction for {patient_id}",
                "confidence": 0.8,
            }

        payload = {
            "id": "chatcmpl-eval",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(content)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 6,
                "total_tokens": 18,
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
def _mock_eval_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EvalMockHandler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def _extract_patient_id(prompt: str) -> str:
    match = re.search(r"patient_id:\s*(p\d+)", prompt)
    if match:
        return str(match.group(1))
    return "p0000"
