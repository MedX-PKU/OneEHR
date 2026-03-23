"""Tests for config schema and loading."""

from pathlib import Path

import pytest


def test_config_defaults():
    from oneehr.config.schema import ExperimentConfig

    cfg = ExperimentConfig()
    assert cfg.task.kind == "binary"
    assert cfg.task.prediction_mode == "patient"
    assert cfg.split.val_size == 0.1
    assert cfg.split.test_size == 0.2
    assert cfg.trainer.lr == 1e-3
    assert cfg.trainer.precision == "fp32"


def test_config_run_dir():
    from oneehr.config.schema import ExperimentConfig, OutputConfig

    cfg = ExperimentConfig(output=OutputConfig(root=Path("/tmp/runs"), run_name="test01"))
    assert cfg.run_dir() == Path("/tmp/runs/test01")


def test_config_task_kinds():
    from oneehr.config.schema import TaskConfig

    for kind in ("binary", "regression", "multiclass", "survival", "multilabel"):
        t = TaskConfig(kind=kind)
        assert t.kind == kind


def test_config_model_params():
    from oneehr.config.schema import ModelConfig

    cfg = ModelConfig(name="xgboost", params={"max_depth": 6, "n_estimators": 100})
    assert cfg.name == "xgboost"
    assert cfg.params["max_depth"] == 6


def test_config_frozen():
    from oneehr.config.schema import TaskConfig

    t = TaskConfig()
    with pytest.raises(AttributeError):
        t.kind = "regression"  # type: ignore[misc]


def test_config_load_toml(tmp_path):
    from oneehr.config.load import load_experiment_config as load_config

    toml_path = tmp_path / "test.toml"
    toml_path.write_text("""
[dataset]
dynamic = "/tmp/dynamic.csv"

[preprocess]
bin_size = "6h"
top_k_codes = 50

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42

[[models]]
name = "xgboost"
[models.params]
max_depth = 3

[trainer]
device = "cpu"
max_epochs = 5

[output]
root = "/tmp/runs"
run_name = "test01"
""")

    cfg = load_config(str(toml_path))
    assert cfg.preprocess.bin_size == "6h"
    assert cfg.preprocess.top_k_codes == 50
    assert cfg.task.kind == "binary"
    assert len(cfg.models) == 1
    assert cfg.models[0].name == "xgboost"
    assert cfg.trainer.max_epochs == 5


def test_trainer_config_defaults():
    from oneehr.config.schema import TrainerConfig

    t = TrainerConfig()
    assert t.early_stopping is True
    assert t.patience == 5
    assert t.scheduler == "none"
    assert t.class_weight == "none"
