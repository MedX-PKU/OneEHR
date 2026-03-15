from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.query import (
    compare_cohorts,
    describe_patient_case,
    describe_run,
    list_failure_cases,
    list_runs,
    read_failure_cases,
)


def test_query_list_runs_and_describe_run(tmp_path: Path) -> None:
    run_root, cfg_path = _build_analyzed_run(tmp_path=tmp_path, run_name="query_run", seed=21)

    runs = list_runs(run_root.parent)
    assert len(runs) == 1
    assert runs[0]["run_name"] == "query_run"
    assert runs[0]["has_analysis_index"] is True

    desc = describe_run(run_root)
    assert desc["run_name"] == "query_run"
    assert desc["manifest"]["task"]["kind"] == "binary"
    assert desc["training"]["models"] == ["xgboost"]
    assert {item["name"] for item in desc["analysis"]["modules"]} >= {"cohort_analysis", "prediction_audit"}

    payload = _run_json(
        ["oneehr", "query", "runs", "describe", "--config", str(cfg_path)],
        cwd=Path.cwd(),
    )
    assert payload["query"] == "runs.describe"
    assert payload["run"]["run_name"] == "query_run"


def test_query_prompt_registry() -> None:
    payload = _run_json(
        ["oneehr", "query", "prompts", "list", "--family", "prediction"],
        cwd=Path.cwd(),
    )
    names = {item["name"] for item in payload["templates"]}
    assert "summary_v1" in names

    desc = _run_json(
        ["oneehr", "query", "prompts", "describe", "--template", "summary_v1"],
        cwd=Path.cwd(),
    )
    assert desc["template"]["family"] == "prediction"
    assert "output_schema" in desc["template"]["default_sections"]


def test_query_analysis_cases_and_cohorts(tmp_path: Path) -> None:
    run_root, _ = _build_analyzed_run(tmp_path=tmp_path, run_name="query_contracts", seed=9)

    modules = _run_json(
        ["oneehr", "query", "analysis", "modules", "--run-dir", str(run_root)],
        cwd=Path.cwd(),
    )
    assert set(modules["modules"]) >= {"dataset_profile", "cohort_analysis", "prediction_audit"}

    summary = _run_json(
        [
            "oneehr",
            "query",
            "analysis",
            "summary",
            "--run-dir",
            str(run_root),
            "--module",
            "prediction_audit",
        ],
        cwd=Path.cwd(),
    )
    assert summary["summary"]["module"] == "prediction_audit"

    table = _run_json(
        [
            "oneehr",
            "query",
            "analysis",
            "table",
            "--run-dir",
            str(run_root),
            "--module",
            "prediction_audit",
            "--table",
            "slices",
            "--limit",
            "1",
        ],
        cwd=Path.cwd(),
    )
    assert table["table"]["row_count"] == 1
    assert set(table["table"]["columns"]) >= {"model", "split", "error_rate"}

    cases = _run_json(
        [
            "oneehr",
            "query",
            "analysis",
            "failures",
            "--run-dir",
            str(run_root),
        ],
        cwd=Path.cwd(),
    )
    assert len(cases["cases"]) > 0
    case_name = cases["cases"][0]["name"]

    case_rows = read_failure_cases(run_root, name=case_name, limit=1)
    patient_id = str(case_rows["records"][0]["patient_id"])

    patient = _run_json(
        [
            "oneehr",
            "query",
            "analysis",
            "patient-case",
            "--run-dir",
            str(run_root),
            "--patient-id",
            patient_id,
            "--limit",
            "1",
        ],
        cwd=Path.cwd(),
    )
    assert patient["patient"]["patient_id"] == patient_id
    assert patient["patient"]["n_matches"] >= 1

    cohort = _run_json(
        [
            "oneehr",
            "query",
            "cohorts",
            "compare",
            "--run-dir",
            str(run_root),
            "--split",
            "fold0",
            "--left-role",
            "train",
            "--right-role",
            "test",
            "--top-k",
            "3",
        ],
        cwd=Path.cwd(),
    )
    assert cohort["comparison"]["split"] == "fold0"
    assert cohort["comparison"]["feature_drift_available"] is True
    assert len(cohort["comparison"]["top_feature_drift"]) <= 3


def test_query_helper_functions(tmp_path: Path) -> None:
    run_root, _ = _build_analyzed_run(tmp_path=tmp_path, run_name="query_helpers", seed=13)

    failure_sets = list_failure_cases(run_root)
    assert len(failure_sets) > 0

    case_name = failure_sets[0]["name"]
    rows = read_failure_cases(run_root, name=case_name, limit=2)
    assert rows["row_count"] >= 1

    patient_id = str(rows["records"][0]["patient_id"])
    desc = describe_patient_case(run_root, patient_id, limit=2)
    assert desc["patient_id"] == patient_id
    assert desc["n_matches"] >= 1

    cohort = compare_cohorts(run_root, split="fold0", left_role="train", right_role="test", top_k=5)
    assert cohort["feature_drift_available"] is True
    assert cohort["left"]["role"] == "train"
    assert cohort["right"]["role"] == "test"


def _run_json(argv: list[str], *, cwd: Path) -> dict[str, object]:
    out = subprocess.check_output(argv, cwd=cwd, text=True)
    return json.loads(out)


def _build_analyzed_run(*, tmp_path: Path, run_name: str, seed: int) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dynamic = _make_simulated_dynamic_events(n_patients=40, seed=seed)
    dynamic_csv = tmp_path / "dynamic.csv"
    _write_dynamic_csv(dynamic_csv, dynamic)

    label_fn = tmp_path / "labels_patient.py"
    _write_patient_label_fn(label_fn)

    cfg = tmp_path / "exp.toml"
    out_root = tmp_path / "runs"
    _write_experiment_toml(
        path=cfg,
        dynamic_csv=dynamic_csv,
        label_fn_ref=f"{label_fn}:build_labels",
        out_root=out_root,
        run_name=run_name,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg), "--force"])
    subprocess.check_call(
        [
            "oneehr",
            "analyze",
            "--config",
            str(cfg),
            "--module",
            "dataset_profile",
            "--module",
            "cohort_analysis",
            "--module",
            "prediction_audit",
        ]
    )
    return out_root / run_name, cfg


def _write_dynamic_csv(path: Path, df: pd.DataFrame) -> None:
    df[["patient_id", "event_time", "code", "value"]].to_csv(path, index=False)


def _make_simulated_dynamic_events(*, n_patients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    rows: list[dict[str, object]] = []
    for pid in range(n_patients):
        patient_id = f"p{pid:04d}"
        for day in range(2):
            t0 = base + pd.Timedelta(days=day)
            rows.append({"patient_id": patient_id, "event_time": t0, "code": "LAB_A", "value": float(rng.normal())})
            rows.append({"patient_id": patient_id, "event_time": t0, "code": "LAB_B", "value": float(rng.normal())})
            rows.append({"patient_id": patient_id, "event_time": t0, "code": "MED_X", "value": float(rng.integers(0, 2))})
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
                "    m = df.groupby('patient_id', sort=True)['value'].mean()",
                "    return pd.DataFrame({'patient_id': m.index.astype(str), 'label': (m.to_numpy() > 0).astype(int)})",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_experiment_toml(
    *,
    path: Path,
    dynamic_csv: Path,
    label_fn_ref: str,
    out_root: Path,
    run_name: str,
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
                "top_k_codes = 50",
                "min_code_count = 1",
                "",
                "[task]",
                'kind = "binary"',
                'prediction_mode = "patient"',
                "",
                "[labels]",
                f'fn = "{label_fn_ref}"',
                "bin_from_time_col = true",
                "",
                "[split]",
                'kind = "kfold"',
                "seed = 0",
                "n_splits = 2",
                "val_size = 0.2",
                "test_size = 0.2",
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
                'final_refit = "train_val"',
                'final_model_source = "refit"',
                "bootstrap_test = false",
                "bootstrap_n = 10",
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
