from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.cases import list_cases, read_case
from oneehr.config.load import load_experiment_config
from oneehr.data.splits import load_splits
from oneehr.query import (
    collect_case_evidence,
    get_case_predictions,
    get_case_static,
    get_case_timeline,
    render_case_prompt,
)
from oneehr.workspace import WorkspaceStore, open_run_workspace
from test_review import _build_review_run, _mock_review_server


def test_cases_cli_and_query_primitives(tmp_path: Path) -> None:
    run_root, cfg_path = _build_cases_run(tmp_path=tmp_path, run_name="cases_run", seed=17)

    cases = list_cases(run_root)
    assert len(cases) > 0
    case_id = str(cases[0]["case_id"])

    case = read_case(run_root, case_id, limit=3)
    assert case["case_id"] == case_id
    assert len(case["events"]) >= 1
    assert "artifacts" in case

    timeline = get_case_timeline(run_root, case_id, limit=2)
    assert timeline["row_count"] == 2

    static_payload = get_case_static(run_root, case_id)
    assert static_payload["feature_count"] >= 2
    assert "age" in static_payload["features"]

    preds = get_case_predictions(run_root, case_id)
    assert preds["row_count"] >= 1
    assert preds["records"][0]["origin"] == "model"
    assert preds["records"][0]["predictor_name"] == "xgboost"

    evidence = collect_case_evidence(run_root, case_id, limit=2)
    assert evidence["case"]["case_id"] == case_id
    assert len(evidence["timeline"]["records"]) == 2
    assert evidence["analysis_refs"]["modules"]

    cfg = load_experiment_config(cfg_path)
    prompt = render_case_prompt(cfg=cfg, run_root=run_root, case_id=case_id)
    assert prompt["template"] == "summary_v1"
    assert "Prediction Task" in prompt["prompt"]

    cli_cases = _run_json(
        ["oneehr", "query", "cases", "list", "--run-dir", str(run_root), "--limit", "1"],
        cwd=Path.cwd(),
    )
    assert cli_cases["query"] == "cases.list"
    assert len(cli_cases["cases"]) == 1

    cli_evidence = _run_json(
        [
            "oneehr",
            "query",
            "cases",
            "evidence",
            "--run-dir",
            str(run_root),
            "--case-id",
            case_id,
            "--limit",
            "1",
        ],
        cwd=Path.cwd(),
    )
    assert cli_evidence["evidence"]["timeline"]["row_count"] == 1

    cli_prompt = _run_json(
        [
            "oneehr",
            "query",
            "cases",
            "render-prompt",
            "--config",
            str(cfg_path),
            "--run-dir",
            str(run_root),
            "--case-id",
            case_id,
            "--template",
            "summary_v1",
        ],
        cwd=Path.cwd(),
    )
    assert cli_prompt["prompt"]["family"] == "prediction"
    assert "Patient Profile" in cli_prompt["prompt"]["prompt"]


def test_cases_respect_saved_test_splits(tmp_path: Path) -> None:
    run_root, _ = _build_cases_run(tmp_path=tmp_path, run_name="cases_split_guard", seed=23)
    splits = load_splits(run_root / "splits")
    assert splits

    cases = list_cases(run_root)
    split_to_test = {sp.name: set(sp.test_patients.astype(str).tolist()) for sp in splits}
    for row in cases:
        assert str(row["patient_id"]) in split_to_test[str(row["split"])]


def test_workspace_domain_unifies_run_case_and_analysis_reads(tmp_path: Path) -> None:
    run_root, _ = _build_cases_run(tmp_path=tmp_path, run_name="workspace_domain", seed=29)

    store = WorkspaceStore(run_root.parent)
    runs = store.list_runs()
    assert runs[0]["run_name"] == "workspace_domain"
    assert runs[0]["has_cases_index"] is True

    workspace = open_run_workspace(run_root)
    desc = workspace.describe()
    assert desc["cases"]["case_count"] >= 1
    assert desc["analysis"]["has_index"] is True
    assert "prediction_audit" in workspace.analysis_modules()

    cases = workspace.case_records(limit=1)
    assert len(cases) == 1
    case_id = str(cases[0]["case_id"])
    case = workspace.read_case(case_id, limit=2)
    assert case["case_id"] == case_id
    assert len(case["events"]) == 2

    artifacts = workspace.failure_case_artifacts("prediction_audit")
    assert len(artifacts) >= 1
    patient_id = str(cases[0]["patient_id"])
    patient_matches = workspace.patient_case_matches(patient_id, "prediction_audit", limit=2)
    assert patient_matches["patient_id"] == patient_id


def test_workspace_domain_reads_agent_detail_artifacts(tmp_path: Path) -> None:
    with _mock_review_server() as (_, base_url):
        run_root, cfg_path = _build_review_run(
            tmp_path=tmp_path,
            run_name="workspace_agent_domain",
            seed=37,
            base_url=base_url,
        )
        subprocess.check_call(
            ["oneehr", "agent", "review", "--config", str(cfg_path)],
            env={**os.environ, "TEST_OPENAI_API_KEY": "dummy"},
        )

    workspace = open_run_workspace(run_root)
    assert workspace.agent_task_actors("review") == ["mock-review"]
    assert workspace.agent_task_splits("review") == ["split0"]
    assert workspace.agent_task_detail_artifacts("review")

    detail_rows = workspace.agent_task_detail_rows("review", actor="mock-review", parsed_ok=True)
    assert not detail_rows.empty
    assert "review_summary" in detail_rows.columns
    assert set(detail_rows["reviewer_name"].astype(str)) == {"mock-review"}

    failure_rows = workspace.agent_task_failure_rows("review", actor="mock-review")
    assert failure_rows.empty


def _run_json(argv: list[str], *, cwd: Path) -> dict[str, object]:
    out = subprocess.check_output(argv, cwd=cwd, text=True)
    return json.loads(out)


def _build_cases_run(*, tmp_path: Path, run_name: str, seed: int) -> tuple[Path, Path]:
    dynamic = _make_simulated_dynamic_events(n_patients=24, seed=seed)
    static = _make_static_table(n_patients=24, seed=seed)

    dynamic_csv = tmp_path / f"{run_name}_dynamic.csv"
    static_csv = tmp_path / f"{run_name}_static.csv"
    label_fn = tmp_path / f"{run_name}_labels.py"
    cfg_path = tmp_path / f"{run_name}.toml"
    out_root = tmp_path / "runs"

    dynamic.to_csv(dynamic_csv, index=False)
    static.to_csv(static_csv, index=False)
    _write_patient_label_fn(label_fn)
    _write_cases_experiment_toml(
        path=cfg_path,
        dynamic_csv=dynamic_csv,
        static_csv=static_csv,
        label_fn_ref=f"{label_fn}:build_labels",
        out_root=out_root,
        run_name=run_name,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg_path)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg_path), "--force"])
    subprocess.check_call(
        [
            "oneehr",
            "analyze",
            "--config",
            str(cfg_path),
            "--module",
            "dataset_profile",
            "--module",
            "cohort_analysis",
            "--module",
            "prediction_audit",
        ]
    )
    subprocess.check_call(["oneehr", "cases", "build", "--config", str(cfg_path), "--force"])
    return out_root / run_name, cfg_path


def _make_simulated_dynamic_events(*, n_patients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    rows: list[dict[str, object]] = []
    for pid in range(n_patients):
        patient_id = f"p{pid:04d}"
        for day in range(3):
            ts = base + pd.Timedelta(days=day)
            rows.append({"patient_id": patient_id, "event_time": ts, "code": "LAB_A", "value": float(rng.normal())})
            rows.append({"patient_id": patient_id, "event_time": ts, "code": "LAB_B", "value": float(rng.normal())})
            rows.append({"patient_id": patient_id, "event_time": ts, "code": "MED_X", "value": float(rng.integers(0, 2))})
    return pd.DataFrame(rows)


def _make_static_table(*, n_patients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    rows = []
    for pid in range(n_patients):
        rows.append(
            {
                "patient_id": f"p{pid:04d}",
                "age": int(30 + (pid % 20)),
                "sex": "M" if pid % 2 == 0 else "F",
                "bmi": float(20 + rng.normal()),
            }
        )
    return pd.DataFrame(rows)


def _write_patient_label_fn(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import pandas as pd",
                "",
                "",
                "def build_labels(dynamic: pd.DataFrame, static, label, cfg):",
                "    df = dynamic[dynamic['code'].astype(str) == 'LAB_A'].copy()",
                "    df['patient_id'] = df['patient_id'].astype(str)",
                "    df['value'] = df['value'].astype(float)",
                "    means = df.groupby('patient_id', sort=True)['value'].mean()",
                "    return pd.DataFrame({'patient_id': means.index.astype(str), 'label': (means.to_numpy() > 0).astype(int)})",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_cases_experiment_toml(
    *,
    path: Path,
    dynamic_csv: Path,
    static_csv: Path,
    label_fn_ref: str,
    out_root: Path,
    run_name: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "[dataset]",
                f'dynamic = "{dynamic_csv}"',
                f'static = "{static_csv}"',
                "",
                "[preprocess]",
                'bin_size = "1d"',
                'numeric_strategy = "mean"',
                'categorical_strategy = "onehot"',
                'code_selection = "frequency"',
                "top_k_codes = 50",
                "",
                "[task]",
                'kind = "binary"',
                'prediction_mode = "patient"',
                "",
                "[labels]",
                f'fn = "{label_fn_ref}"',
                "",
                "[split]",
                'kind = "kfold"',
                "seed = 0",
                "n_splits = 2",
                "val_size = 0.2",
                "",
                "[model]",
                'name = "xgboost"',
                "",
                "[trainer]",
                'device = "cpu"',
                "seed = 123",
                "max_epochs = 1",
                "batch_size = 16",
                "lr = 1e-3",
                "early_stopping = false",
                "",
                "[cases]",
                "include_static = true",
                "include_analysis_refs = true",
                "max_events = 5",
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
