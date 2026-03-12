from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.agent import (
    collect_case_evidence,
    get_case_predictions,
    get_patient_static,
    get_patient_timeline,
    list_workspace_cases,
    read_workspace_case,
    render_case_prompt,
)
from oneehr.config.load import load_experiment_config
from oneehr.data.splits import load_splits


def test_workspace_cli_and_task_primitives(tmp_path: Path) -> None:
    run_root, cfg_path = _build_workspace_run(tmp_path=tmp_path, run_name="workspace_run", seed=17)

    cases = list_workspace_cases(run_root)
    assert len(cases) > 0
    case_id = str(cases[0]["case_id"])

    case = read_workspace_case(run_root, case_id, limit=3)
    assert case["case_id"] == case_id
    assert len(case["events"]) >= 1
    assert "artifacts" in case

    timeline = get_patient_timeline(run_root, case_id, limit=2)
    assert timeline["row_count"] == 2

    static_payload = get_patient_static(run_root, case_id)
    assert static_payload["feature_count"] >= 2
    assert "age" in static_payload["features"]

    preds = get_case_predictions(run_root, case_id)
    assert preds["row_count"] >= 1
    assert preds["records"][0]["source"] == "train"

    evidence = collect_case_evidence(run_root, case_id, limit=2)
    assert evidence["workspace"]["case_id"] == case_id
    assert len(evidence["timeline"]["records"]) == 2
    assert evidence["analysis_refs"]["modules"]

    cfg = load_experiment_config(cfg_path)
    prompt = render_case_prompt(cfg=cfg, run_root=run_root, case_id=case_id)
    assert prompt["template"] == "summary_v1"
    assert "Prediction Task" in prompt["prompt"]

    cli_cases = _run_json(
        ["oneehr", "inspect", "--tool", "workspace.list_cases", "--run-dir", str(run_root), "--limit", "1"],
        cwd=Path.cwd(),
    )
    assert len(cli_cases["cases"]) == 1

    cli_evidence = _run_json(
        [
            "oneehr",
            "inspect",
            "--tool",
            "tasks.collect_evidence",
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
            "inspect",
            "--tool",
            "tasks.render_prompt",
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


def test_workspace_cases_respect_saved_test_splits(tmp_path: Path) -> None:
    run_root, _ = _build_workspace_run(tmp_path=tmp_path, run_name="workspace_split_guard", seed=23)
    splits = load_splits(run_root / "splits")
    assert splits

    cases = list_workspace_cases(run_root)
    split_to_test = {sp.name: set(sp.test_patients.astype(str).tolist()) for sp in splits}
    for row in cases:
        assert str(row["patient_id"]) in split_to_test[str(row["split"])]


def _run_json(argv: list[str], *, cwd: Path) -> dict[str, object]:
    out = subprocess.check_output(argv, cwd=cwd, text=True)
    return json.loads(out)


def _build_workspace_run(*, tmp_path: Path, run_name: str, seed: int) -> tuple[Path, Path]:
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
    _write_workspace_experiment_toml(
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
            "--format",
            "json",
            "--format",
            "csv",
        ]
    )
    subprocess.check_call(["oneehr", "workspace", "--config", str(cfg_path), "--force"])
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


def _write_workspace_experiment_toml(
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
                "[llm]",
                "enabled = false",
                'sample_unit = "patient"',
                'prompt_template = "summary_v1"',
                "json_schema_version = 1",
                "",
                "[llm.prompt]",
                "include_static = true",
                "max_events = 10",
                'time_order = "asc"',
                "",
                "[llm.output]",
                "include_explanation = true",
                "include_confidence = false",
                "",
                "[workspace]",
                "include_static = true",
                "include_analysis_refs = true",
                "max_events = 5",
                'time_order = "asc"',
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
