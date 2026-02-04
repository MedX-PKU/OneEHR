import pandas as pd
import pytest

from oneehr.artifacts.read import RunManifest
from oneehr.artifacts.run_io import RunIO


def test_load_binned_validates_schema(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    # Minimal binned with required + feature column.
    df = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "bin_time": pd.to_datetime(["2020-01-01"]),
            "num__A": [1.0],
        }
    )
    df.to_parquet(run_root / "binned.parquet", index=False)

    manifest = RunManifest(
        schema_version=2,
        data={"features": {"dynamic": {"feature_columns": ["num__A"]}}, "artifacts": {"binned_parquet_path": "binned.parquet"}},
    )
    run = RunIO(run_root=run_root)
    out = run.load_binned(manifest)
    assert list(out.columns) == ["patient_id", "bin_time", "num__A"]


def test_load_binned_missing_feature_raises(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    df = pd.DataFrame({"patient_id": ["p1"], "bin_time": pd.to_datetime(["2020-01-01"])})
    df.to_parquet(run_root / "binned.parquet", index=False)
    manifest = RunManifest(
        schema_version=2,
        data={"features": {"dynamic": {"feature_columns": ["num__A"]}}, "artifacts": {"binned_parquet_path": "binned.parquet"}},
    )
    run = RunIO(run_root=run_root)
    with pytest.raises(SystemExit):
        run.load_binned(manifest)

