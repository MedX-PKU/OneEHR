from __future__ import annotations

import json
from pathlib import Path

from oneehr.artifacts.run_manifest import write_run_manifest
from oneehr.config.schema import DatasetConfig, DynamicTableConfig, ExperimentConfig, ModelConfig, SplitConfig, TaskConfig


def test_write_run_manifest(tmp_path: Path):
    cfg = ExperimentConfig(
        dataset=DatasetConfig(dynamic=DynamicTableConfig(path=Path("x.csv"))),
        task=TaskConfig(kind="binary", prediction_mode="patient"),
        split=SplitConfig(kind="kfold"),
        model=ModelConfig(name="xgboost"),
    )

    out = tmp_path / "run"
    write_run_manifest(
        out_root=out,
        cfg=cfg,
        dynamic_feature_columns=["num__A", "cat__DX__X"],
        static_feature_columns=["num__age", "cat__sex__M"],
        static_postprocess_pipeline=[],
        patient_tabular_path="views/patient_tabular.parquet",
        time_tabular_path=None,
    )

    data = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 5
    assert data["features"]["dynamic"]["feature_columns"] == ["num__A", "cat__DX__X"]
    assert "feature_columns_json_path" not in data["features"]["dynamic"]
    assert "feature_columns_json_path" not in data["features"]["static"]
    assert (data["artifacts"] or {}).get("binned_parquet_path") == "binned.parquet"
    assert data["cases"]["include_static"] is True
    assert data["cases"]["max_events"] == 200
    assert data["agent"]["predict"]["enabled"] is False
    assert data["agent"]["predict"]["prompt_template"] == "summary_v1"
    assert data["agent"]["review"]["enabled"] is False
    assert data["agent"]["review"]["prediction_origins"] == ["model", "agent"]
