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

from oneehr.config.load import load_experiment_config
from oneehr.review.schema import parse_review_response


def test_load_config_with_review_sections(tmp_path: Path) -> None:
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
                "[review]",
                "enabled = true",
                'prompt_template = "evidence_review_v1"',
                'prediction_sources = ["train"]',
                "",
                "[review.prompt]",
                "include_ground_truth = true",
                'time_order = "desc"',
                "",
                "[[review_models]]",
                'name = "mock-review"',
                'provider = "openai_compatible"',
                'base_url = "http://127.0.0.1:9999/v1"',
                'model = "mock-review-model"',
                'api_key_env = "TEST_OPENAI_API_KEY"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(cfg_path)
    assert cfg.review.enabled is True
    assert cfg.review.prompt_template == "evidence_review_v1"
    assert cfg.review.prediction_sources == ["train"]
    assert cfg.review.prompt.time_order == "desc"
    assert cfg.review_models[0].name == "mock-review"


def test_parse_review_response() -> None:
    parsed = parse_review_response(
        '{"supported": true, "clinically_grounded": true, "leakage_suspected": false, '
        '"needs_human_review": false, "overall_score": 0.9, "review_summary": "Supported.", '
        '"key_evidence": ["lab trend"], "missing_evidence": []}'
    )
    assert parsed.parsed_ok is True
    assert parsed.supported is True
    assert parsed.overall_score == 0.9
    assert parsed.key_evidence == ["lab trend"]


def test_llm_review_cli_e2e(tmp_path: Path) -> None:
    with _mock_review_server() as (server, base_url):
        run_root, cfg_path = _build_review_run(tmp_path=tmp_path, run_name="review_run", seed=31, base_url=base_url)
        env = os.environ.copy()
        env["TEST_OPENAI_API_KEY"] = "dummy"

        subprocess.check_call(["oneehr", "llm-review", "--config", str(cfg_path)], env=env)

        summary_path = run_root / "review" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert len(summary["records"]) >= 1
        record = summary["records"][0]
        assert record["review_model"] == "mock-review"
        assert record["target_source"] == "train"
        assert record["parse_success_rate"] == 1.0
        assert record["metrics"]["supported_rate"] == 1.0

        parsed_files = sorted((run_root / "review" / "parsed" / "mock-review").glob("*.parquet"))
        assert parsed_files
        parsed = pd.read_parquet(parsed_files[0])
        assert parsed["parsed_ok"].all()
        assert parsed["supported"].all()

        case_id = json.loads((run_root / "workspace" / "index.json").read_text(encoding="utf-8"))["records"][0]["case_id"]
        prompt_payload = _run_json(
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
                "evidence_review_v1",
                "--source",
                "train",
                "--model-name",
                "xgboost",
            ],
            cwd=Path.cwd(),
        )
        assert prompt_payload["prompt"]["family"] == "review"
        assert "Review Rubric" in prompt_payload["prompt"]["prompt"]

        review_payload = _run_json(
            ["oneehr", "inspect", "--tool", "reviews.read_summary", "--run-dir", str(run_root)],
            cwd=Path.cwd(),
        )
        assert len(review_payload["summary"]["records"]) >= 1
        assert len(server.requests) > 0


class _MockReviewHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(body)  # type: ignore[attr-defined]

        payload = {
            "id": "chatcmpl-review",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "supported": True,
                                "clinically_grounded": True,
                                "leakage_suspected": False,
                                "needs_human_review": False,
                                "overall_score": 0.9,
                                "review_summary": "Evidence supports the prediction.",
                                "key_evidence": ["Recent LAB_A trend aligns with target prediction."],
                                "missing_evidence": [],
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
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
def _mock_review_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockReviewHandler)
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


def _run_json(argv: list[str], *, cwd: Path) -> dict[str, object]:
    out = subprocess.check_output(argv, cwd=cwd, text=True)
    return json.loads(out)


def _build_review_run(*, tmp_path: Path, run_name: str, seed: int, base_url: str) -> tuple[Path, Path]:
    dynamic = _make_simulated_dynamic_events(n_patients=20, seed=seed)
    static = _make_static_table(n_patients=20, seed=seed)

    dynamic_csv = tmp_path / f"{run_name}_dynamic.csv"
    static_csv = tmp_path / f"{run_name}_static.csv"
    label_fn = tmp_path / f"{run_name}_labels.py"
    cfg_path = tmp_path / f"{run_name}.toml"
    out_root = tmp_path / "runs"

    dynamic.to_csv(dynamic_csv, index=False)
    static.to_csv(static_csv, index=False)
    _write_patient_label_fn(label_fn)
    _write_review_experiment_toml(
        path=cfg_path,
        dynamic_csv=dynamic_csv,
        static_csv=static_csv,
        label_fn_ref=f"{label_fn}:build_labels",
        out_root=out_root,
        run_name=run_name,
        base_url=base_url,
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
                "age": int(35 + (pid % 15)),
                "sex": "M" if pid % 2 == 0 else "F",
                "bmi": float(21 + rng.normal()),
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


def _write_review_experiment_toml(
    *,
    path: Path,
    dynamic_csv: Path,
    static_csv: Path,
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
                "[workspace]",
                "include_static = true",
                "include_analysis_refs = true",
                "max_events = 5",
                "",
                "[review]",
                "enabled = true",
                'prompt_template = "evidence_review_v1"',
                'prediction_sources = ["train"]',
                "concurrency = 1",
                "max_retries = 1",
                "timeout_seconds = 5.0",
                "temperature = 0.0",
                "top_p = 1.0",
                "",
                "[review.prompt]",
                "include_static = true",
                "include_ground_truth = true",
                "include_analysis_context = true",
                "max_events = 5",
                'time_order = "asc"',
                "",
                "[[review_models]]",
                'name = "mock-review"',
                'provider = "openai_compatible"',
                f'base_url = "{base_url}"',
                'model = "mock-review-model"',
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
