from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from oneehr.web.service import WebUIService
from support_runs import build_analyzed_run, build_eval_run, build_trained_run


def test_webui_service_run_dashboards_and_drilldowns(tmp_path: Path) -> None:
    run_root, _ = build_analyzed_run(tmp_path=tmp_path, run_name="webui_run", seed=31)
    service = WebUIService(root_dir=run_root.parent)

    runs = service.list_runs_payload()
    assert runs["run_count"] == 1
    assert runs["runs"][0]["run_name"] == "webui_run"
    assert runs["runs"][0]["analysis_status"] == "ready"

    desc = service.describe_run_payload(run_name="webui_run")
    assert desc["hero"]["analysis_module_count"] >= 3
    assert desc["run"]["training"]["models"] == ["xgboost"]
    assert "navigation" in desc
    assert "cases_route" not in desc["navigation"]
    assert "agents_route" not in desc["navigation"]

    dashboard = service.analysis_dashboard_payload(run_name="webui_run", module_name="prediction_audit")
    assert dashboard["module"]["name"] == "prediction_audit"
    assert dashboard["module"]["summary"]["status"] == "ok"
    assert any(chart["id"] == "model_primary_metric" for chart in dashboard["charts"])
    assert any(table["name"] == "slices" for table in dashboard["tables"])
    assert dashboard["drilldowns"]["patient_case_supported"] is True
    assert len(dashboard["drilldowns"]["case_artifacts"]) > 0


def test_webui_service_eval_payload(tmp_path: Path) -> None:
    run_root, _ = build_eval_run(tmp_path=tmp_path, run_name="webui_eval", seed=19)
    service = WebUIService(root_dir=run_root.parent)

    runs = service.list_runs_payload()
    assert runs["runs"][0]["eval_status"] == "ready"

    desc = service.describe_run_payload(run_name="webui_eval")
    assert desc["hero"]["eval_instance_count"] == 4
    assert desc["hero"]["eval_system_count"] == 8
    assert desc["navigation"]["eval_route"] == f"/runs/{run_root.name}/eval"

    payload = service.eval_payload(run_name="webui_eval")
    assert payload["status"] == "ok"
    assert payload["index"]["instance_count"] == 4
    assert payload["report"]["primary_metric"] == "accuracy"
    assert any(table["name"] == "leaderboard" for table in payload["tables"])

    table = service.eval_table_payload(
        run_name="webui_eval",
        table_name="leaderboard",
        limit=2,
        offset=0,
        sort_by="accuracy",
        sort_dir="desc",
    )
    assert table["table"] == "leaderboard"
    assert table["row_count"] == 2

    instance_id = str(payload["index"]["records"][0]["instance_id"])
    instance = service.eval_instance_payload(run_name="webui_eval", instance_id=instance_id)
    assert instance["instance"]["instance_id"] == instance_id

    trace = service.eval_trace_payload(
        run_name="webui_eval",
        system_name="healthcareagent_eval",
        stage="plan",
        limit=5,
    )
    assert trace["table"] == "trace_rows"
    assert trace["row_count"] >= 1


def test_webui_service_testing_payloads(tmp_path: Path) -> None:
    run_root, cfg = build_trained_run(tmp_path=tmp_path, run_name="webui_test_audit", seed=27)
    subprocess.check_call(["oneehr", "test", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "analyze", "--config", str(cfg), "--module", "test_audit"])

    service = WebUIService(root_dir=run_root.parent)
    desc = service.describe_run_payload(run_name="webui_test_audit")
    assert desc["run"]["testing"]["record_count"] >= 1
    assert desc["hero"]["test_record_count"] >= 1

    dashboard = service.analysis_dashboard_payload(run_name="webui_test_audit", module_name="test_audit")
    assert dashboard["module"]["name"] == "test_audit"
    assert dashboard["module"]["summary"]["status"] == "ok"
    assert any(chart["id"] == "model_primary_metric" for chart in dashboard["charts"])
    assert any(table["name"] == "metric_summary" for table in dashboard["tables"])


def test_webui_service_comparison_payload(tmp_path: Path) -> None:
    run_root_a, cfg_a = build_trained_run(tmp_path=tmp_path / "run_a", run_name="cmp_webui_a", seed=5)
    run_root_b, cfg_b = build_trained_run(tmp_path=tmp_path / "run_b", run_name="cmp_webui_b", seed=8)
    subprocess.check_call(["oneehr", "test", "--config", str(cfg_a), "--force"])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg_b), "--force"])
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
    assert comparison["summary"]["test_delta_rows"] > 0
    assert any(table["name"] == "train_metrics" for table in comparison["tables"])
    assert any(table["name"] == "test_metrics" for table in comparison["tables"])
    assert len(comparison["charts"]) >= 1

    dashboard = service.analysis_dashboard_payload(run_name="cmp_webui_a", module_name="prediction_audit")
    assert dashboard["comparison_available"] is True

    table = service.comparison_table_payload(
        run_name="cmp_webui_a",
        table_name="train_metrics",
        limit=1,
        offset=0,
        sort_by="delta_mean",
        sort_dir="desc",
    )
    assert table["row_count"] == 1
    assert table["total_rows"] >= 1
    assert table["table"] == "train_metrics"


def test_webui_service_cohort_compare_payload(tmp_path: Path) -> None:
    run_root, _ = build_analyzed_run(tmp_path=tmp_path, run_name="webui_cohort_compare", seed=23)
    service = WebUIService(root_dir=run_root.parent)

    payload = service.cohort_compare_payload(
        run_name="webui_cohort_compare",
        split="fold0",
        left_role="train",
        right_role="test",
        top_k=4,
    )
    assert payload["comparison"]["split"] == "fold0"
    assert payload["comparison"]["left_role"] == "train"
    assert payload["comparison"]["right_role"] == "test"
    assert payload["comparison"]["feature_drift_available"] is True
    assert len(payload["comparison"]["top_feature_drift"]) <= 4


def test_webui_fastapi_routes(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from oneehr.web import create_app

    run_root, _ = build_eval_run(tmp_path=tmp_path, run_name="webui_api", seed=17)
    client = TestClient(create_app(root_dir=run_root.parent))

    runs = client.get("/api/v1/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["run_name"] == "webui_api"

    eval_payload = client.get(f"/api/v1/runs/{run_root.name}/eval")
    assert eval_payload.status_code == 200
    assert eval_payload.json()["status"] == "ok"

    table = client.get(
        f"/api/v1/runs/{run_root.name}/eval/tables/leaderboard",
        params={"limit": 1, "sort_by": "accuracy", "sort_dir": "desc"},
    )
    assert table.status_code == 200
    assert table.json()["row_count"] == 1
    assert table.json()["table"] == "leaderboard"

    instance_id = str(eval_payload.json()["index"]["records"][0]["instance_id"])
    instance = client.get(f"/api/v1/runs/{run_root.name}/eval/instances/{instance_id}")
    assert instance.status_code == 200
    assert instance.json()["instance"]["instance_id"] == instance_id

    trace = client.get(
        f"/api/v1/runs/{run_root.name}/eval/traces/healthcareagent_eval",
        params={"stage": "plan", "limit": 5},
    )
    assert trace.status_code == 200
    assert trace.json()["row_count"] >= 1

    legacy_cases = client.get(f"/api/v1/runs/{run_root.name}/cases")
    assert legacy_cases.status_code == 404

    legacy_agents = client.get(f"/api/v1/runs/{run_root.name}/agents")
    assert legacy_agents.status_code == 404
