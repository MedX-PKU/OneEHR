from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from oneehr.web.service import WebUIService
from test_analysis import _build_trained_run
from test_inspect import _build_analyzed_run
from test_review import _build_review_run, _mock_review_server
from test_runview import _build_cases_run


def test_webui_service_run_dashboards_and_drilldowns(tmp_path: Path) -> None:
    run_root, _ = _build_analyzed_run(tmp_path=tmp_path, run_name="webui_run", seed=31)
    service = WebUIService(root_dir=run_root.parent)

    runs = service.list_runs_payload()
    assert runs["run_count"] == 1
    assert runs["runs"][0]["run_name"] == "webui_run"
    assert runs["runs"][0]["analysis_status"] == "ready"

    desc = service.describe_run_payload(run_name="webui_run")
    assert desc["hero"]["analysis_module_count"] >= 3
    assert desc["run"]["training"]["models"] == ["xgboost"]
    assert "navigation" in desc
    assert "workspace" not in desc

    dashboard = service.analysis_dashboard_payload(run_name="webui_run", module_name="prediction_audit")
    assert dashboard["module"]["name"] == "prediction_audit"
    assert dashboard["module"]["summary"]["status"] == "ok"
    assert any(chart["id"] == "model_primary_metric" for chart in dashboard["charts"])
    assert any(table["name"] == "slices" for table in dashboard["tables"])
    assert dashboard["drilldowns"]["patient_case_supported"] is True
    assert len(dashboard["drilldowns"]["case_artifacts"]) > 0

    table = service.analysis_table_payload(
        run_name="webui_run",
        module_name="prediction_audit",
        table_name="slices",
        limit=1,
        offset=0,
        sort_by="error_rate",
        sort_dir="desc",
    )
    assert table["row_count"] == 1
    assert table["total_rows"] >= 1

    case_name = dashboard["drilldowns"]["case_artifacts"][0]["name"]
    case_rows = service.analysis_case_rows_payload(
        run_name="webui_run",
        module_name="prediction_audit",
        case_name=case_name,
        limit=5,
        offset=0,
        filter_col="patient_id",
        filter_value="p0",
    )
    assert case_rows["row_count"] >= 1

    patient_id = str(case_rows["records"][0]["patient_id"])
    patient = service.analysis_patient_case_payload(
        run_name="webui_run",
        module_name="prediction_audit",
        patient_id=patient_id,
        limit=1,
    )
    assert patient["patient"]["patient_id"] == patient_id
    assert patient["patient"]["n_matches"] >= 1


def test_webui_service_comparison_payload(tmp_path: Path) -> None:
    run_root_a, cfg_a = _build_trained_run(tmp_path=tmp_path / "run_a", run_name="cmp_webui_a", seed=5)
    run_root_b, _ = _build_trained_run(tmp_path=tmp_path / "run_b", run_name="cmp_webui_b", seed=8)
    subprocess.check_call(
        [
            "oneehr",
            "analyze",
            "--config",
            str(cfg_a),
            "--module",
            "prediction_audit",
            "--compare-run",
            str(run_root_b),
        ]
    )

    service = WebUIService(root_dir=run_root_a.parent)
    comparison = service.comparison_payload(run_name="cmp_webui_a")
    assert comparison["status"] == "ok"
    assert comparison["summary"]["train_delta_rows"] > 0
    assert any(table["name"] == "train_metrics" for table in comparison["tables"])
    assert len(comparison["charts"]) >= 1


def test_webui_service_cases_payloads(tmp_path: Path) -> None:
    run_root, _ = _build_cases_run(tmp_path=tmp_path, run_name="webui_cases", seed=19)
    service = WebUIService(root_dir=run_root.parent)

    cases = service.cases_payload(run_name="webui_cases", limit=10)
    assert cases["status"] == "ok"
    assert cases["case_count"] >= 1
    assert cases["row_count"] >= 1

    case_id = str(cases["records"][0]["case_id"])
    detail = service.case_detail_payload(run_name="webui_cases", case_id=case_id, limit=3)
    assert detail["case"]["case_id"] == case_id
    assert detail["timeline"]["row_count"] >= 1
    assert detail["predictions"]["row_count"] >= 1
    assert detail["static"]["feature_count"] >= 1

    split_name = str(cases["records"][0]["split"])
    filtered = service.cases_payload(run_name="webui_cases", split=split_name, limit=10)
    assert filtered["row_count"] >= 1
    assert split_name in filtered["splits"]
    assert all(str(record["split"]) == split_name for record in filtered["records"])


def test_webui_service_agents_payload(tmp_path: Path) -> None:
    with _mock_review_server() as (_, base_url):
        run_root, cfg_path = _build_review_run(
            tmp_path=tmp_path,
            run_name="webui_agents",
            seed=41,
            base_url=base_url,
        )
        env = os.environ.copy()
        env["TEST_OPENAI_API_KEY"] = "dummy"
        subprocess.check_call(["oneehr", "agent", "review", "--config", str(cfg_path)], env=env)

    service = WebUIService(root_dir=run_root.parent)
    payload = service.agents_payload(run_name="webui_agents")
    assert payload["predict"]["status"] == "missing"
    assert payload["review"]["status"] == "ok"
    assert payload["review"]["table"]["row_count"] >= 1
    assert payload["review"]["detail_available"] is True

    records = service.agent_records_payload(run_name="webui_agents", task_name="review", actor="mock-review", limit=5)
    assert records["status"] == "ok"
    assert records["row_count"] >= 1
    assert any(column["name"] == "review_summary" for column in records["columns"])

    failures = service.agent_failures_payload(run_name="webui_agents", task_name="review", actor="mock-review")
    assert failures["status"] == "ok"
    assert failures["row_count"] == 0

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from oneehr.web import create_app

    client = TestClient(create_app(root_dir=run_root.parent))
    route = client.get(
        f"/api/v1/runs/{run_root.name}/agents/review/records",
        params={"actor": "mock-review", "limit": 5},
    )
    assert route.status_code == 200
    assert route.json()["row_count"] >= 1


def test_webui_fastapi_routes(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from oneehr.web import create_app

    run_root, _ = _build_cases_run(tmp_path=tmp_path, run_name="webui_api", seed=17)
    client = TestClient(create_app(root_dir=run_root.parent))

    runs = client.get("/api/v1/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["run_name"] == "webui_api"

    dashboard = client.get(f"/api/v1/runs/{run_root.name}/analysis/prediction_audit/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["module"]["name"] == "prediction_audit"

    table = client.get(
        f"/api/v1/runs/{run_root.name}/analysis/prediction_audit/tables/slices",
        params={"limit": 1, "sort_by": "error_rate", "sort_dir": "desc"},
    )
    assert table.status_code == 200
    assert table.json()["row_count"] == 1
    assert table.json()["total_rows"] >= 1

    cases = client.get(f"/api/v1/runs/{run_root.name}/analysis/prediction_audit/cases")
    assert cases.status_code == 200
    assert len(cases.json()["case_artifacts"]) > 0

    cases_response = client.get(f"/api/v1/runs/{run_root.name}/cases")
    assert cases_response.status_code == 200
    assert cases_response.json()["case_count"] >= 1

    split_name = str(cases_response.json()["records"][0]["split"])
    filtered_cases = client.get(
        f"/api/v1/runs/{run_root.name}/cases",
        params={"split": split_name, "limit": 1},
    )
    assert filtered_cases.status_code == 200
    assert filtered_cases.json()["row_count"] == 1
    assert split_name in filtered_cases.json()["splits"]

    case_id = cases_response.json()["records"][0]["case_id"]
    case_detail = client.get(f"/api/v1/runs/{run_root.name}/cases/{case_id}")
    assert case_detail.status_code == 200
    assert case_detail.json()["case"]["case_id"] == case_id

    agents = client.get(f"/api/v1/runs/{run_root.name}/agents")
    assert agents.status_code == 200
    assert agents.json()["predict"]["status"] == "missing"


def test_cli_webui_help() -> None:
    out = subprocess.check_output(["oneehr", "webui", "serve", "--help"], text=True)
    assert "--frontend-dist" in out
