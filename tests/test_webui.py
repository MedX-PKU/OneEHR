from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from oneehr.web.service import WebUIService
from test_analysis import _build_trained_run
from test_inspect import _build_analyzed_run


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


def test_webui_fastapi_routes(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from oneehr.web import create_app

    run_root, _ = _build_analyzed_run(tmp_path=tmp_path, run_name="webui_api", seed=17)
    client = TestClient(create_app(root_dir=run_root.parent))

    runs = client.get("/api/v1/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["run_name"] == "webui_api"

    dashboard = client.get(f"/api/v1/runs/{run_root.name}/analysis/prediction_audit/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["module"]["name"] == "prediction_audit"

    cases = client.get(f"/api/v1/runs/{run_root.name}/analysis/prediction_audit/cases")
    assert cases.status_code == 200
    assert len(cases.json()["case_artifacts"]) > 0


def test_cli_webui_help() -> None:
    out = subprocess.check_output(["oneehr", "webui", "serve", "--help"], text=True)
    assert "--frontend-dist" in out
