from __future__ import annotations

import json
import subprocess
from pathlib import Path

from oneehr.runview import RunCatalog, open_run_view
from support_runs import build_eval_run, build_trained_run


def test_run_view_describe_exposes_eval_first_contract(tmp_path: Path) -> None:
    run_root, _ = build_eval_run(tmp_path=tmp_path, run_name="runview_eval", seed=29)

    catalog = RunCatalog(run_root.parent)
    runs = catalog.list_runs()
    assert runs[0]["run_name"] == "runview_eval"
    assert runs[0]["has_eval_index"] is True
    assert runs[0]["has_eval_report_summary"] is True
    assert "has_cases_index" not in runs[0]
    assert "has_agent_predict_summary" not in runs[0]

    run_view = open_run_view(run_root)
    desc = run_view.describe()
    assert desc["eval"]["instance_count"] == 4
    assert desc["eval"]["system_count"] == 8
    assert desc["eval"]["leaderboard_rows"] == 8
    assert desc["artifacts"]["has_eval_dir"] is True
    assert "cases" not in desc
    assert "agent_predict" not in desc
    assert "agent_review" not in desc

    leaderboard = run_view.eval_table("leaderboard")
    assert not leaderboard.empty
    assert "system_name" in leaderboard.columns

    index = run_view.eval_index()
    summary = run_view.eval_summary()
    report = run_view.eval_report_summary()
    assert index["instance_count"] == 4
    assert len(summary["records"]) == 8
    assert report["primary_metric"] == "accuracy"


def test_run_view_describe_includes_testing_summary(tmp_path: Path) -> None:
    run_root, cfg_path = build_trained_run(tmp_path=tmp_path, run_name="runview_testing", seed=33)
    subprocess.check_call(["oneehr", "test", "--config", str(cfg_path), "--force"])

    catalog = RunCatalog(run_root.parent)
    runs = catalog.list_runs()
    assert runs[0]["has_test_summary"] is True

    run_view = open_run_view(run_root)
    desc = run_view.describe()
    assert desc["testing"]["record_count"] >= 1
    assert desc["testing"]["summary_path"] == "test_runs/test_summary.json"
    assert desc["testing"]["best_model"] is not None
    assert desc["testing"]["best_model"]["metric"] == "auroc"


def test_query_runs_describe_matches_eval_first_shape(tmp_path: Path) -> None:
    run_root, cfg_path = build_eval_run(tmp_path=tmp_path, run_name="runview_query", seed=17)

    payload = json.loads(
        subprocess.check_output(
            ["oneehr", "query", "runs", "describe", "--config", str(cfg_path)],
            text=True,
        )
    )
    run = payload["run"]
    assert run["run_name"] == run_root.name
    assert run["eval"]["instance_count"] == 4
    assert "cases" not in run
    assert "agent_predict" not in run
