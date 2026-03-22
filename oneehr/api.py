"""OneEHR Python API — programmatic access to the pipeline.

Usage::

    import oneehr

    config = oneehr.load_config("experiment.toml")
    oneehr.preprocess(config)
    oneehr.train(config)
    oneehr.test(config)
    results = oneehr.analyze(config)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oneehr.config.load import load_experiment_config
from oneehr.config.schema import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a TOML file."""
    return load_experiment_config(path)


@dataclass
class PreprocessResult:
    run_dir: Path
    n_patients: int
    n_features: int
    feature_columns: list[str]


def preprocess(config: ExperimentConfig | str | Path) -> PreprocessResult:
    """Run the preprocessing pipeline.

    Parameters
    ----------
    config : ExperimentConfig, str, or Path
        Either a loaded config object or a path to a TOML file.
    """
    cfg = _resolve_config(config)

    from oneehr.artifacts.manifest import read_manifest
    from oneehr.artifacts.materialize import materialize_preprocess_artifacts
    from oneehr.data.io import load_dynamic_table_optional, load_label_table, load_static_table

    dynamic = load_dynamic_table_optional(cfg.dataset.dynamic)
    static = load_static_table(cfg.dataset.static)
    label = load_label_table(cfg.dataset.label)
    run_dir = cfg.run_dir()

    materialize_preprocess_artifacts(
        dynamic=dynamic, static=static, label=label, cfg=cfg, run_dir=run_dir,
    )

    manifest = read_manifest(run_dir)
    feat_cols = manifest["feature_columns"]

    import pandas as pd
    binned = pd.read_parquet(run_dir / "preprocess" / "binned.parquet")
    n_patients = binned["patient_id"].nunique()

    return PreprocessResult(
        run_dir=run_dir,
        n_patients=n_patients,
        n_features=len(feat_cols),
        feature_columns=feat_cols,
    )


@dataclass
class TrainResult:
    run_dir: Path
    models_trained: list[str]


def train(config: ExperimentConfig | str | Path, *, force: bool = False) -> TrainResult:
    """Run model training.

    Parameters
    ----------
    config : ExperimentConfig, str, or Path
    force : overwrite existing train artifacts
    """
    cfg = _resolve_config(config)
    cfg_path = _config_path(config)

    from oneehr.cli.train import run_train
    run_train(str(cfg_path), force)

    model_names = [m.name for m in cfg.models]
    return TrainResult(run_dir=cfg.run_dir(), models_trained=model_names)


@dataclass
class TestResult:
    run_dir: Path
    predictions_path: Path
    metrics_path: Path


def test(config: ExperimentConfig | str | Path, *, force: bool = False) -> TestResult:
    """Run model evaluation on the test set.

    Parameters
    ----------
    config : ExperimentConfig, str, or Path
    force : overwrite existing test artifacts
    """
    cfg = _resolve_config(config)
    cfg_path = _config_path(config)

    from oneehr.cli.test import run_test
    run_test(str(cfg_path), force)

    test_dir = cfg.run_dir() / "test"
    return TestResult(
        run_dir=cfg.run_dir(),
        predictions_path=test_dir / "predictions.parquet",
        metrics_path=test_dir / "metrics.json",
    )


@dataclass
class AnalyzeResult:
    run_dir: Path
    modules_run: list[str]
    results: dict[str, Any]


def analyze(
    config: ExperimentConfig | str | Path,
    *,
    module: str | None = None,
) -> AnalyzeResult:
    """Run analysis modules.

    Parameters
    ----------
    config : ExperimentConfig, str, or Path
    module : specific module to run, or None for all
    """
    cfg = _resolve_config(config)
    cfg_path = _config_path(config)

    from oneehr.cli.analyze import run_analyze
    run_analyze(str(cfg_path), module=module)

    # Load results
    import json
    analyze_dir = cfg.run_dir() / "analyze"
    results = {}
    modules_run = []
    if analyze_dir.exists():
        for f in sorted(analyze_dir.glob("*.json")):
            name = f.stem
            modules_run.append(name)
            results[name] = json.loads(f.read_text())

    return AnalyzeResult(
        run_dir=cfg.run_dir(),
        modules_run=modules_run,
        results=results,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_config(config: ExperimentConfig | str | Path) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return config
    return load_experiment_config(config)


def _config_path(config: ExperimentConfig | str | Path) -> Path:
    """Get path for CLI functions that need a file path."""
    if isinstance(config, (str, Path)):
        return Path(config)
    # For ExperimentConfig objects, we can't recover the path.
    # The CLI functions need the path, so this is a limitation.
    raise TypeError(
        "When using ExperimentConfig objects directly, call the underlying "
        "CLI functions or pass a config file path instead."
    )
