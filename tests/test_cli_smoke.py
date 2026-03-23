"""Smoke tests for the new 4-command CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _make_dynamic(tmp_path: Path, n_patients: int = 40, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    rows = []
    for pid in range(n_patients):
        patient_id = f"p{pid:04d}"
        for day in range(2):
            t = base + pd.Timedelta(days=day)
            rows.append({"patient_id": patient_id, "event_time": t, "code": "LAB_A", "value": float(rng.normal())})
            rows.append({"patient_id": patient_id, "event_time": t, "code": "LAB_B", "value": float(rng.normal())})
    path = tmp_path / "dynamic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_label(tmp_path: Path, n_patients: int = 40) -> Path:
    rows = []
    for pid in range(n_patients):
        rows.append(
            {
                "patient_id": f"p{pid:04d}",
                "label_time": "2020-01-02",
                "label_code": "outcome",
                "label_value": pid % 2,  # balanced binary labels
            }
        )
    path = tmp_path / "label.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_config(tmp_path: Path, dynamic_csv: Path, label_csv: Path, out_root: Path) -> Path:
    cfg = tmp_path / "exp.toml"
    cfg.write_text(
        f"""
[dataset]
dynamic = "{dynamic_csv}"
label = "{label_csv}"

[preprocess]
bin_size = "1d"
top_k_codes = 10

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42
val_size = 0.2
test_size = 0.2

[[models]]
name = "xgboost"
[models.params]
max_depth = 3
n_estimators = 10

[trainer]
device = "cpu"
seed = 42
max_epochs = 1
batch_size = 16

[output]
root = "{out_root}"
run_name = "test_run"
""",
        encoding="utf-8",
    )
    return cfg


def test_preprocess_writes_artifacts(tmp_path: Path) -> None:
    dynamic_csv = _make_dynamic(tmp_path)
    label_csv = _make_label(tmp_path)
    out_root = tmp_path / "runs"
    cfg = _write_config(tmp_path, dynamic_csv, label_csv, out_root)

    from oneehr.cli.main import main

    main(["preprocess", "--config", str(cfg)])

    run_dir = out_root / "test_run"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "preprocess" / "binned.parquet").exists()
    assert (run_dir / "preprocess" / "split.json").exists()

    # Check split.json structure
    split_data = json.loads((run_dir / "preprocess" / "split.json").read_text())
    assert "train" in split_data
    assert "val" in split_data
    assert "test" in split_data
