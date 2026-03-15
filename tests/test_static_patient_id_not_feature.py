from __future__ import annotations

from pathlib import Path

import pandas as pd

from oneehr.artifacts.materialize import materialize_preprocess_artifacts
from oneehr.artifacts.read import read_run_manifest
from oneehr.config.schema import (
    DatasetConfig,
    DynamicTableConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    PreprocessConfig,
    SplitConfig,
    StaticTableConfig,
    TaskConfig,
)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_static_patient_id_not_encoded_as_feature(tmp_path: Path) -> None:
    # Minimal dynamic table needed for preprocessing.
    dynamic = pd.DataFrame(
        {
            "patient_id": ["1", "2", "3"],
            "event_time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
            "code": ["lab_a", "lab_a", "lab_a"],
            "value": [1.0, 2.0, 3.0],
        }
    )

    # Static has patient_id as numeric (common when read from CSV without dtype hints).
    static = pd.DataFrame(
        {
            "patient_id": [1.0, 2.0, 3.0],
            "Age": [50.0, 60.0, 70.0],
            "Sex": [1.0, 0.0, 1.0],
        }
    )

    dyn_path = tmp_path / "dynamic.csv"
    stat_path = tmp_path / "static.csv"
    _write_csv(dyn_path, dynamic.assign(event_time=dynamic["event_time"].dt.strftime("%Y-%m-%d")))
    _write_csv(stat_path, static)

    cfg = ExperimentConfig(
        dataset=DatasetConfig(
            dynamic=DynamicTableConfig(path=dyn_path),
            static=StaticTableConfig(path=stat_path),
            label=None,
        ),
        datasets=None,
        preprocess=PreprocessConfig(
            bin_size="1d",
            code_selection="all",
            min_code_count=1,
            top_k_codes=None,
            numeric_strategy="mean",
            categorical_strategy="onehot",
            pipeline=[],
        ),
        task=TaskConfig(kind="binary", prediction_mode="patient"),
        split=SplitConfig(kind="random", n_splits=2, val_size=0.2, test_size=0.0, seed=1, time_boundary=None),
        output=OutputConfig(root=tmp_path, run_name="run"),
        model=ModelConfig(name="xgboost"),
    )

    out_root = tmp_path / "out"
    materialize_preprocess_artifacts(dynamic=dynamic, static=static, label=None, cfg=cfg, out_root=out_root)

    manifest = read_run_manifest(out_root)
    assert manifest is not None
    cols = manifest.static_feature_columns()
    assert all("patient_id" not in c for c in cols)
    assert not (out_root / "features" / "static" / "feature_columns.json").exists()
    assert not (out_root / "labels_meta.json").exists()
