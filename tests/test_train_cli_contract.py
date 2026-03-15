from __future__ import annotations

from pathlib import Path

import pytest

from oneehr.artifacts.run_manifest import write_run_manifest
from oneehr.cli.train import run_train
from oneehr.config.schema import (
    DatasetConfig,
    DynamicTableConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
    TaskConfig,
)


def test_run_train_allows_existing_preprocess_run_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path, _ = _prepare_preprocessed_run(tmp_path)
    called: list[tuple[str, bool]] = []

    monkeypatch.setattr(
        "oneehr.cli.train._run_benchmark",
        lambda cfg_path_arg, force=False: called.append((cfg_path_arg, force)),
    )

    run_train(str(cfg_path), force=False)

    assert called == [(str(cfg_path), False)]


def test_run_train_requires_force_when_training_outputs_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path, run_root = _prepare_preprocessed_run(tmp_path)
    (run_root / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("oneehr.cli.train._run_benchmark", lambda cfg_path_arg, force=False: None)

    with pytest.raises(SystemExit, match="Training artifacts already exist"):
        run_train(str(cfg_path), force=False)


def _prepare_preprocessed_run(tmp_path: Path) -> tuple[Path, Path]:
    dynamic_path = tmp_path / "dynamic.csv"
    dynamic_path.write_text("patient_id,event_time,code,value\np1,2020-01-01,LAB_A,1.0\n", encoding="utf-8")

    cfg = ExperimentConfig(
        dataset=DatasetConfig(dynamic=DynamicTableConfig(path=dynamic_path)),
        task=TaskConfig(kind="binary", prediction_mode="patient"),
        split=SplitConfig(kind="random"),
        model=ModelConfig(name="xgboost"),
        output=OutputConfig(root=tmp_path / "runs", run_name="example"),
    )
    cfg_path = tmp_path / "exp.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[dataset]",
                f'dynamic = "{dynamic_path}"',
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
                "[output]",
                f'root = "{cfg.output.root}"',
                f'run_name = "{cfg.output.run_name}"',
            ]
        ),
        encoding="utf-8",
    )

    run_root = cfg.output.root / cfg.output.run_name
    write_run_manifest(
        out_root=run_root,
        cfg=cfg,
        dynamic_feature_columns=["num__A"],
        static_feature_columns=[],
        static_postprocess_pipeline=None,
        patient_tabular_path="views/patient_tabular.parquet",
        time_tabular_path=None,
    )
    (run_root / "splits").mkdir(parents=True, exist_ok=True)
    (run_root / "splits" / "split0.json").write_text(
        '{"name":"split0","train_patients":["p1"],"val_patients":[],"test_patients":["p2"]}',
        encoding="utf-8",
    )
    return cfg_path, run_root
