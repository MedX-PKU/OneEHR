"""End-to-end tests for the new 4-command pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _make_dynamic(tmp_path: Path, n_patients: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    rows = []
    for pid in range(n_patients):
        patient_id = f"p{pid:04d}"
        for day in range(3):
            t = base + pd.Timedelta(days=day)
            rows.append({"patient_id": patient_id, "event_time": t, "code": "LAB_A", "value": float(rng.normal())})
            rows.append({"patient_id": patient_id, "event_time": t, "code": "LAB_B", "value": float(rng.normal())})
            rows.append({"patient_id": patient_id, "event_time": t, "code": "MED_X", "value": float(rng.integers(0, 2))})
    path = tmp_path / "dynamic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_dynamic_with_missing(tmp_path: Path, n_patients: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    rows = []
    for pid in range(n_patients):
        patient_id = f"p{pid:04d}"
        for day in range(3):
            t = base + pd.Timedelta(days=day)
            rows.append({"patient_id": patient_id, "event_time": t, "code": "LAB_A", "value": float(rng.normal())})
            if (pid + day) % 2 == 0:
                rows.append({"patient_id": patient_id, "event_time": t, "code": "LAB_B", "value": float(rng.normal())})
            if day != 1 or pid % 3 != 0:
                rows.append({"patient_id": patient_id, "event_time": t, "code": "MED_X", "value": float(rng.integers(0, 2))})
    path = tmp_path / "dynamic_missing.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_label(tmp_path: Path, n_patients: int = 60) -> Path:
    rows = []
    for pid in range(n_patients):
        rows.append(
            {
                "patient_id": f"p{pid:04d}",
                "label_time": "2020-01-03",
                "label_code": "outcome",
                "label_value": pid % 2,
            }
        )
    path = tmp_path / "label.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_config(
    tmp_path: Path,
    dynamic_csv: Path,
    label_csv: Path,
    out_root: Path,
    *,
    run_name: str = "e2e",
    models: list[str] | None = None,
) -> Path:
    models = models or ["xgboost"]
    model_blocks = "\n".join([f'[[models]]\nname = "{m}"\n[models.params]\n' for m in models])

    cfg = tmp_path / "exp.toml"
    cfg.write_text(
        f"""
[dataset]
dynamic = "{dynamic_csv}"
label = "{label_csv}"

[preprocess]
bin_size = "1d"
top_k_codes = 20

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42
val_size = 0.2
test_size = 0.2

{model_blocks}

[trainer]
device = "cpu"
seed = 42
max_epochs = 1
batch_size = 16
early_stopping = false

[output]
root = "{out_root}"
run_name = "{run_name}"
""",
        encoding="utf-8",
    )
    return cfg


def test_full_pipeline_xgboost(tmp_path: Path) -> None:
    """preprocess -> train -> test -> analyze with XGBoost."""
    dynamic_csv = _make_dynamic(tmp_path)
    label_csv = _make_label(tmp_path)
    out_root = tmp_path / "runs"
    cfg = _write_config(tmp_path, dynamic_csv, label_csv, out_root, models=["xgboost"])

    from oneehr.cli.main import main

    main(["preprocess", "--config", str(cfg)])

    run_dir = out_root / "e2e"
    assert (run_dir / "preprocess" / "binned.parquet").exists()
    assert (run_dir / "preprocess" / "split.json").exists()
    assert (run_dir / "manifest.json").exists()

    main(["train", "--config", str(cfg)])

    assert (run_dir / "train" / "xgboost" / "checkpoint.ckpt").exists()
    assert (run_dir / "train" / "xgboost" / "meta.json").exists()

    main(["test", "--config", str(cfg)])

    preds = run_dir / "test" / "predictions.parquet"
    assert preds.exists()
    df = pd.read_parquet(preds)
    assert "system" in df.columns
    assert "patient_id" in df.columns
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert (df["system"] == "xgboost").any()

    metrics_path = run_dir / "test" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "systems" in metrics
    assert len(metrics["systems"]) > 0

    main(["analyze", "--config", str(cfg)])

    assert (run_dir / "analyze" / "comparison.json").exists()


def test_full_pipeline_pai(tmp_path: Path) -> None:
    dynamic_csv = _make_dynamic_with_missing(tmp_path)
    label_csv = _make_label(tmp_path)
    out_root = tmp_path / "runs"
    cfg = tmp_path / "pai.toml"
    cfg.write_text(
        f"""
[dataset]
dynamic = "{dynamic_csv}"
label = "{label_csv}"

[preprocess]
bin_size = "1d"
top_k_codes = 20

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42
val_size = 0.2
test_size = 0.2

[[models]]
name = "pai"
[models.params]
hidden_dim = 8
prompt_init = "median"

[trainer]
device = "cpu"
seed = 42
max_epochs = 1
batch_size = 16
early_stopping = false

[output]
root = "{out_root}"
run_name = "pai_e2e"
""",
        encoding="utf-8",
    )

    from oneehr.cli.main import main

    main(["preprocess", "--config", str(cfg)])
    run_dir = out_root / "pai_e2e"
    assert (run_dir / "preprocess" / "obs_mask.parquet").exists()

    main(["train", "--config", str(cfg)])
    assert (run_dir / "train" / "pai" / "checkpoint.ckpt").exists()

    main(["test", "--config", str(cfg)])
    preds = pd.read_parquet(run_dir / "test" / "predictions.parquet")
    assert (preds["system"] == "pai").any()


def test_full_pipeline_grud(tmp_path: Path) -> None:
    dynamic_csv = _make_dynamic_with_missing(tmp_path)
    label_csv = _make_label(tmp_path)
    out_root = tmp_path / "runs"
    cfg = tmp_path / "grud.toml"
    cfg.write_text(
        f"""
[dataset]
dynamic = "{dynamic_csv}"
label = "{label_csv}"

[preprocess]
bin_size = "1d"
top_k_codes = 20

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42
val_size = 0.2
test_size = 0.2

[[models]]
name = "grud"
[models.params]
hidden_dim = 8

[trainer]
device = "cpu"
seed = 42
max_epochs = 1
batch_size = 16
early_stopping = false

[output]
root = "{out_root}"
run_name = "grud_e2e"
""",
        encoding="utf-8",
    )

    from oneehr.cli.main import main

    main(["preprocess", "--config", str(cfg)])
    run_dir = out_root / "grud_e2e"
    assert (run_dir / "preprocess" / "obs_mask.parquet").exists()

    main(["train", "--config", str(cfg)])
    assert (run_dir / "train" / "grud" / "checkpoint.ckpt").exists()

    main(["test", "--config", str(cfg)])
    preds = pd.read_parquet(run_dir / "test" / "predictions.parquet")
    assert (preds["system"] == "grud").any()


def test_full_pipeline_raindrop(tmp_path: Path) -> None:
    dynamic_csv = _make_dynamic_with_missing(tmp_path)
    label_csv = _make_label(tmp_path)
    out_root = tmp_path / "runs"
    cfg = tmp_path / "raindrop.toml"
    cfg.write_text(
        f"""
[dataset]
dynamic = "{dynamic_csv}"
label = "{label_csv}"

[preprocess]
bin_size = "1d"
top_k_codes = 20

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42
val_size = 0.2
test_size = 0.2

[[models]]
name = "raindrop"
[models.params]
hidden_dim = 8

[trainer]
device = "cpu"
seed = 42
max_epochs = 1
batch_size = 16
early_stopping = false

[output]
root = "{out_root}"
run_name = "raindrop_e2e"
""",
        encoding="utf-8",
    )

    from oneehr.cli.main import main

    main(["preprocess", "--config", str(cfg)])
    run_dir = out_root / "raindrop_e2e"
    assert (run_dir / "preprocess" / "obs_mask.parquet").exists()

    main(["train", "--config", str(cfg)])
    assert (run_dir / "train" / "raindrop" / "checkpoint.ckpt").exists()

    main(["test", "--config", str(cfg)])
    preds = pd.read_parquet(run_dir / "test" / "predictions.parquet")
    assert (preds["system"] == "raindrop").any()


def test_full_pipeline_graphcare(tmp_path: Path) -> None:
    dynamic_csv = _make_dynamic(tmp_path)
    label_csv = _make_label(tmp_path)
    out_root = tmp_path / "runs"
    cfg = tmp_path / "graphcare.toml"
    cfg.write_text(
        f"""
[dataset]
dynamic = "{dynamic_csv}"
label = "{label_csv}"

[preprocess]
bin_size = "1d"
top_k_codes = 20

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42
val_size = 0.2
test_size = 0.2

[[models]]
name = "graphcare"
[models.params]
hidden_dim = 8
kg_source = "lightweight"
kg_top_k = 4
kg_min_cooccurrence = 1

[trainer]
device = "cpu"
seed = 42
max_epochs = 1
batch_size = 16
early_stopping = false

[output]
root = "{out_root}"
run_name = "graphcare_e2e"
""",
        encoding="utf-8",
    )

    from oneehr.cli.main import main

    main(["preprocess", "--config", str(cfg)])
    run_dir = out_root / "graphcare_e2e"
    assert (run_dir / "preprocess" / "feature_schema.json").exists()

    main(["train", "--config", str(cfg)])
    assert (run_dir / "train" / "graphcare" / "checkpoint.ckpt").exists()

    main(["test", "--config", str(cfg)])
    preds = pd.read_parquet(run_dir / "test" / "predictions.parquet")
    assert (preds["system"] == "graphcare").any()
