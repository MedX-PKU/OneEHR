from __future__ import annotations

import json
import os
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.analysis import list_analysis_modules, read_analysis_index, read_analysis_summary, read_analysis_table
from oneehr.config.load import load_experiment_config


def test_load_config_with_analysis_section(tmp_path: Path) -> None:
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
                "[model]",
                'name = "xgboost"',
                "",
                "[analysis]",
                'default_modules = ["dataset_profile", "prediction_audit"]',
                "top_k = 15",
                'stratify_by = ["sex"]',
                "case_limit = 10",
                "save_plot_specs = true",
                "shap_max_samples = 64",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(cfg_path)
    assert cfg.analysis.default_modules == ["dataset_profile", "prediction_audit"]
    assert cfg.analysis.top_k == 15
    assert cfg.analysis.stratify_by == ["sex"]
    assert cfg.analysis.case_limit == 10
    assert cfg.analysis.shap_max_samples == 64


def test_cli_analyze_default_writes_structured_outputs(tmp_path: Path) -> None:
    run_root, cfg = _build_trained_run(tmp_path=tmp_path, run_name="analysis_default", seed=17)

    subprocess.check_call(["oneehr", "analyze", "--config", str(cfg)])

    analysis_root = run_root / "analysis"
    index = read_analysis_index(run_root)
    modules = list_analysis_modules(run_root)
    assert "dataset_profile" in modules
    assert "prediction_audit" in modules
    assert "interpretability" in modules
    assert "agent_audit" in modules

    assert (analysis_root / "index.json").exists()
    assert not (analysis_root / "index.md").exists()
    assert not (analysis_root / "index.html").exists()
    assert (analysis_root / "dataset_profile" / "summary.json").exists()
    assert (analysis_root / "prediction_audit" / "slices.csv").exists()
    assert (analysis_root / "prediction_audit" / "cases").exists()
    assert (analysis_root / "cohort_analysis" / "feature_drift.csv").exists()
    assert (analysis_root / "temporal_analysis" / "segments.csv").exists()
    assert (analysis_root / "agent_audit" / "summary.json").exists()
    assert any(path.name.startswith("feature_importance_xgboost_") for path in analysis_root.iterdir())

    pred_summary = read_analysis_summary(run_root, "prediction_audit")
    assert pred_summary["status"] == "ok"
    pred_table = read_analysis_table(run_root, "prediction_audit", "slices")
    assert not pred_table.empty
    assert set(pred_table.columns) >= {"model", "split", "error_rate"}

    agent_summary = read_analysis_summary(run_root, "agent_audit")
    assert agent_summary["status"] == "skipped"
    assert index["comparison"] is None


def test_cli_analyze_compare_run_writes_comparison(tmp_path: Path) -> None:
    run_root_a, cfg_a = _build_trained_run(tmp_path=tmp_path / "run_a", run_name="cmp_a", seed=5)
    run_root_b, _ = _build_trained_run(tmp_path=tmp_path / "run_b", run_name="cmp_b", seed=8)

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

    comparison_dir = run_root_a / "analysis" / "comparison"
    assert (comparison_dir / "summary.json").exists()
    assert (comparison_dir / "train_metrics.csv").exists()
    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["train_delta_rows"] > 0
    index = read_analysis_index(run_root_a)
    assert index["comparison"]["summary_path"] == "analysis/comparison/summary.json"


def test_cli_analyze_agent_audit_module(tmp_path: Path, monkeypatch) -> None:
    dynamic_csv = tmp_path / "dynamic.csv"
    label_fn = tmp_path / "label_fn.py"
    cfg_path = tmp_path / "exp.toml"
    out_root = tmp_path / "runs"
    run_name = "agent_audit"

    _write_agent_dynamic_csv(dynamic_csv)
    _write_patient_parity_label_fn(label_fn)

    with _mock_chat_server() as (_, base_url):
        _write_agent_experiment_toml(
            path=cfg_path,
            dynamic_csv=dynamic_csv,
            label_fn_ref=f"{label_fn}:build_labels",
            out_root=out_root,
            run_name=run_name,
            base_url=base_url,
        )
        monkeypatch.setenv("TEST_OPENAI_API_KEY", "dummy")
        env = os.environ.copy()
        env["TEST_OPENAI_API_KEY"] = "dummy"

        subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg_path)], env=env)
        subprocess.check_call(["oneehr", "agent", "predict", "--config", str(cfg_path)], env=env)
        subprocess.check_call(
            [
                "oneehr",
                "analyze",
                "--config",
                str(cfg_path),
                "--module",
                "agent_audit",
            ],
            env=env,
        )

    run_root = out_root / run_name
    summary = read_analysis_summary(run_root, "agent_audit")
    assert summary["status"] == "ok"
    slices = read_analysis_table(run_root, "agent_audit", "slices")
    assert not slices.empty
    assert set(slices.columns) >= {"predictor_name", "split", "parse_success_rate", "coverage", "total_tokens"}
    assert (run_root / "analysis" / "agent_audit" / "plots" / "parse_success_rate.json").exists()


def _build_trained_run(*, tmp_path: Path, run_name: str, seed: int) -> tuple[Path, Path]:
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
        task_kind="binary",
        prediction_mode="patient",
        models=["xgboost"],
        split_kind="kfold",
        hpo_enabled=False,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
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
    task_kind: str,
    prediction_mode: str,
    models: list[str],
    split_kind: str,
    hpo_enabled: bool,
) -> None:
    model_blocks = "\n".join([f'[[models]]\nname = "{m}"\n' for m in models])
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
                f'kind = "{task_kind}"',
                f'prediction_mode = "{prediction_mode}"',
                "",
                "[labels]",
                f'fn = "{label_fn_ref}"',
                "bin_from_time_col = true",
                "",
                "[split]",
                f'kind = "{split_kind}"',
                "seed = 0",
                "n_splits = 2",
                "val_size = 0.2",
                "test_size = 0.2",
                "",
                model_blocks.strip(),
                "",
                "[hpo]",
                f"enabled = {str(hpo_enabled).lower()}",
                'metric = "val_loss"',
                'mode = "min"',
                'scope = "single"',
                "grid = []",
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


def _write_agent_dynamic_csv(path: Path) -> None:
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


def _write_agent_experiment_toml(
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
                "[model]",
                'name = "xgboost"',
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


class _MockChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = str(body["messages"][-1]["content"])
        patient_id = (
            prompt.split("patient_id: ", 1)[1].splitlines()[0].strip() if "patient_id: " in prompt else "p0000"
        )
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
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
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
def _mock_chat_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()
