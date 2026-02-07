from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_dynamic_csv(path: Path, df: pd.DataFrame) -> None:
    # Keep the schema doctor-friendly and stable.
    df[["patient_id", "event_time", "code", "value"]].to_csv(path, index=False)


def _make_simulated_dynamic_events(*, n_patients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")

    rows: list[dict[str, object]] = []
    for pid in range(n_patients):
        patient_id = f"p{pid:04d}"
        # Create 2 days worth of events so 1d binning has multiple bins.
        for day in range(2):
            t0 = base + pd.Timedelta(days=day)
            # Numeric lab signal + some noise.
            lab = float(rng.normal(loc=0.0, scale=1.0))
            rows.append({"patient_id": patient_id, "event_time": t0, "code": "LAB_A", "value": lab})
            # Another numeric lab.
            rows.append({"patient_id": patient_id, "event_time": t0, "code": "LAB_B", "value": float(rng.normal())})
            # A sparse numeric "exposure" code (0/1); still categorical-ish but numeric.
            rows.append(
                {"patient_id": patient_id, "event_time": t0, "code": "MED_X", "value": float(rng.integers(0, 2))}
            )
    return pd.DataFrame(rows)


def _write_patient_label_fn(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import numpy as np",
                "import pandas as pd",
                "",
                "",
                "def build_labels(dynamic: pd.DataFrame, static, label, cfg):",
                "    # Deterministic, leakage-safe labels based on the dynamic table only.",
                "    # Patient label = 1 if mean(LAB_A) > 0.",
                "    df = dynamic[dynamic['code'].astype(str) == 'LAB_A'].copy()",
                "    df['patient_id'] = df['patient_id'].astype(str)",
                "    df['value'] = df['value'].astype(float)",
                "    m = df.groupby('patient_id', sort=True)['value'].mean()",
                "    out = pd.DataFrame({'patient_id': m.index.astype(str), 'label': (m.to_numpy() > 0).astype(int)})",
                "    return out",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_stratified_patient_label_fn(path: Path) -> None:
    """Write a label_fn that guarantees class balance by patient_id.

    This avoids folds/splits being skipped for single-class train/val and
    ensures every model produces artifacts + prediction outputs.
    """

    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import pandas as pd",
                "",
                "",
                "def build_labels(dynamic: pd.DataFrame, static, label, cfg):",
                "    df = dynamic[['patient_id']].drop_duplicates().copy()",
                "    df['patient_id'] = df['patient_id'].astype(str)",
                "    # patient_id like 'p0001' -> 1; fallback 0",
                "    # Use modulo of pid_int to ensure stable balance within GroupKFold splits.",
                "    df['pid_int'] = df['patient_id'].str.extract(r'(\\d+)', expand=False).fillna('0').astype(int)",
                "    df['label'] = (df['pid_int'] // 10 % 2).astype(int)",
                "    return df[['patient_id', 'label']]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_time_label_fn(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import pandas as pd",
                "",
                "",
                "def build_labels(dynamic: pd.DataFrame, static, label, cfg):",
                "    # Time-level labels aligned to preprocess binning.",
                "    # label_time is the event_time (will be binned downstream if configured).",
                "    df = dynamic[dynamic['code'].astype(str) == 'LAB_A'].copy()",
                "    df['patient_id'] = df['patient_id'].astype(str)",
                "    df['label_time'] = pd.to_datetime(df['event_time'], errors='raise')",
                "    df['value'] = df['value'].astype(float)",
                "    df['label'] = (df['value'] > 0).astype(int)",
                "    return df[['patient_id', 'label_time', 'label']]",
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
    calibration_enabled: bool = False,
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
                # Use a fixed seed to make tests deterministic and to avoid
                # class-imbalanced val splits causing model training to be skipped.
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
                "final_refit = \"train_val\"",
                "final_model_source = \"refit\"",
                "bootstrap_test = false",
                "bootstrap_n = 10",
                "",
                "[calibration]",
                f"enabled = {str(calibration_enabled).lower()}",
                'method = "temperature"',
                'source = "val"',
                'threshold_strategy = "f1"',
                "use_calibrated = true",
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


def _assert_preds_sane(path: Path, *, task_kind: str) -> None:
    assert path.exists(), f"missing predictions file: {path}"
    df = pd.read_parquet(path)
    assert "y_pred" in df.columns
    y_pred = df["y_pred"].to_numpy(dtype=float)
    assert np.isfinite(y_pred).all()
    if task_kind == "binary":
        assert (y_pred >= 0).all()
        assert (y_pred <= 1).all()


def _assert_test_outputs(run_root: Path) -> None:
    test_runs = run_root / "test_runs"
    assert test_runs.exists()
    assert (test_runs / "test_summary.json").exists()
    # If models are present, we expect some per-split metrics + preds.
    records = json.loads((test_runs / "test_summary.json").read_text(encoding="utf-8")).get("records") or []
    assert isinstance(records, list)
    assert len(records) > 0
    assert any((test_runs / f).exists() for f in test_runs.iterdir() if f.name.startswith("preds_"))


@pytest.mark.parametrize("split_kind", ["kfold", "random"])
def test_cli_e2e_patient_binary_xgboost_calibration(tmp_path: Path, split_kind: str) -> None:
    dynamic = _make_simulated_dynamic_events(n_patients=60, seed=0)
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
        run_name=f"patient_bin_{split_kind}",
        task_kind="binary",
        prediction_mode="patient",
        models=["xgboost"],
        split_kind=split_kind,
        hpo_enabled=False,
        calibration_enabled=True,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg), "--force"])

    run_root = out_root / f"patient_bin_{split_kind}"
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "summary.json").exists()
    assert (run_root / "hpo_best.csv").exists()

    preds = run_root / "preds" / "xgboost" / ("fold0.parquet" if split_kind == "kfold" else "split0.parquet")
    _assert_preds_sane(preds, task_kind="binary")
    _assert_test_outputs(run_root)


def test_cli_e2e_patient_regression_xgboost(tmp_path: Path) -> None:
    dynamic = _make_simulated_dynamic_events(n_patients=80, seed=1)
    # Make regression signal: shift LAB_A to be positive-ish with patient id.
    dynamic.loc[dynamic["code"] == "LAB_A", "value"] = (
        dynamic.loc[dynamic["code"] == "LAB_A", "value"].astype(float)
        + dynamic["patient_id"].astype(str).str.extract(r"p(\\d+)")[0].fillna("0").astype(int) / 1000.0
    )
    dynamic_csv = tmp_path / "dynamic.csv"
    _write_dynamic_csv(dynamic_csv, dynamic)

    label_fn = tmp_path / "labels_patient_reg.py"
    label_fn.write_text(
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
                "    out = pd.DataFrame({'patient_id': m.index.astype(str), 'label': m.to_numpy().astype(float)})",
                "    return out",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = tmp_path / "exp.toml"
    out_root = tmp_path / "runs"
    _write_experiment_toml(
        path=cfg,
        dynamic_csv=dynamic_csv,
        label_fn_ref=f"{label_fn}:build_labels",
        out_root=out_root,
        run_name="patient_reg",
        task_kind="regression",
        prediction_mode="patient",
        models=["xgboost"],
        split_kind="kfold",
        hpo_enabled=False,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg), "--force"])

    run_root = out_root / "patient_reg"
    preds = run_root / "preds" / "xgboost" / "fold0.parquet"
    _assert_preds_sane(preds, task_kind="regression")
    _assert_test_outputs(run_root)


def test_cli_e2e_time_binary_gru(tmp_path: Path) -> None:
    dynamic = _make_simulated_dynamic_events(n_patients=50, seed=2)
    dynamic_csv = tmp_path / "dynamic.csv"
    _write_dynamic_csv(dynamic_csv, dynamic)

    label_fn = tmp_path / "labels_time.py"
    _write_time_label_fn(label_fn)

    cfg = tmp_path / "exp.toml"
    out_root = tmp_path / "runs"
    _write_experiment_toml(
        path=cfg,
        dynamic_csv=dynamic_csv,
        label_fn_ref=f"{label_fn}:build_labels",
        out_root=out_root,
        run_name="time_bin_gru",
        task_kind="binary",
        prediction_mode="time",
        models=["gru"],
        split_kind="kfold",
        hpo_enabled=False,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg), "--force"])

    run_root = out_root / "time_bin_gru"
    # Some DL time-level models can be skipped on tiny simulated datasets
    # (e.g., single-class in a fold). For this E2E test we only require that
    # the pipeline produces the run contract + test outputs directory.
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "summary.json").exists()
    assert (run_root / "hpo_best.csv").exists()
    assert (run_root / "test_runs" / "test_summary.json").exists()


@pytest.mark.parametrize(
    "model_name",
    [
        # Time-level DL models (keep epochs=1 in config writer).
        "tcn",
        "gru",
        "lstm",
        "rnn",
        "transformer",
        "mlp",
        "retain",
        "stagenet",
        "adacare",
        "concare",
        "grasp",
        "mcgru",
        "dragent",
    ],
)
def test_cli_e2e_time_binary_dl_models_smoke(tmp_path: Path, model_name: str) -> None:
    """Broader smoke coverage for DL model registry on time-level tasks.

    This is intentionally light: we assert the run contract exists and the test
    command wrote a summary. Training may legitimately skip a model for a split
    (e.g., single-class train/val on simulated data), so we do not require
    `models/` or `preds/` to exist.
    """

    dynamic = _make_simulated_dynamic_events(n_patients=160, seed=4)
    dynamic_csv = tmp_path / "dynamic.csv"
    _write_dynamic_csv(dynamic_csv, dynamic)

    label_fn = tmp_path / "labels_time.py"
    _write_time_label_fn(label_fn)

    cfg = tmp_path / "exp.toml"
    out_root = tmp_path / "runs"
    _write_experiment_toml(
        path=cfg,
        dynamic_csv=dynamic_csv,
        label_fn_ref=f"{label_fn}:build_labels",
        out_root=out_root,
        run_name=f"time_bin_{model_name}",
        task_kind="binary",
        prediction_mode="time",
        models=[model_name],
        split_kind="kfold",
        hpo_enabled=False,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg), "--force"])

    run_root = out_root / f"time_bin_{model_name}"
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "summary.json").exists()
    assert (run_root / "hpo_best.csv").exists()
    assert (run_root / "test_runs" / "test_summary.json").exists()

    payload = json.loads((run_root / "test_runs" / "test_summary.json").read_text(encoding="utf-8"))
    assert "records" in payload


@pytest.mark.parametrize(
    "model_name",
    [
        # Tabular ML models.
        "xgboost",
        "catboost",
        "rf",
        "dt",
        "gbdt",
        # Patient-level DL models.
        "gru",
        "lstm",
        "rnn",
        "transformer",
        "mlp",
        "retain",
        "stagenet",
        "adacare",
        "concare",
        "grasp",
        "mcgru",
        "dragent",
    ],
)
def test_cli_e2e_patient_binary_all_models_smoke(tmp_path: Path, model_name: str) -> None:
    """Broader smoke coverage for ML+DL models on patient-level tasks."""

    dynamic = _make_simulated_dynamic_events(n_patients=240, seed=5)
    dynamic_csv = tmp_path / "dynamic.csv"
    _write_dynamic_csv(dynamic_csv, dynamic)

    label_fn = tmp_path / "labels_patient_stratified.py"
    _write_stratified_patient_label_fn(label_fn)

    cfg = tmp_path / "exp.toml"
    out_root = tmp_path / "runs"
    _write_experiment_toml(
        path=cfg,
        dynamic_csv=dynamic_csv,
        label_fn_ref=f"{label_fn}:build_labels",
        out_root=out_root,
        run_name=f"patient_bin_{model_name}",
        task_kind="binary",
        prediction_mode="patient",
        models=[model_name],
        split_kind="kfold",
        hpo_enabled=False,
        calibration_enabled=(model_name in {"xgboost", "catboost"}),
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg), "--force"])

    run_root = out_root / f"patient_bin_{model_name}"
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "summary.json").exists()
    assert (run_root / "hpo_best.csv").exists()
    test_summary = run_root / "test_runs" / "test_summary.json"
    assert test_summary.exists()
    payload = json.loads(test_summary.read_text(encoding="utf-8"))
    assert "records" in payload
    # With stratified labels, training should not skip; require preds for the model.
    preds_dir = run_root / "preds" / model_name
    assert preds_dir.exists()
    assert any(p.suffix == ".parquet" for p in preds_dir.iterdir())


def test_cli_analyze_writes_feature_importance(tmp_path: Path) -> None:
    dynamic = _make_simulated_dynamic_events(n_patients=60, seed=3)
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
        run_name="analyze_run",
        task_kind="binary",
        prediction_mode="patient",
        models=["xgboost"],
        split_kind="kfold",
        hpo_enabled=False,
    )

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg), "--force"])
    subprocess.check_call(["oneehr", "analyze", "--config", str(cfg), "--method", "xgboost"])

    run_root = out_root / "analyze_run"
    analysis_dir = run_root / "analysis"
    assert analysis_dir.exists()
    assert any(p.name.startswith("feature_importance_xgboost_") for p in analysis_dir.iterdir())
