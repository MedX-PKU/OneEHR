import pandas as pd

from pathlib import Path

from oneehr.config.schema import DatasetConfig, PreprocessConfig
from oneehr.data.binning import bin_events


def _events_df():
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p2"],
            "event_time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:10:00", "2020-01-02 00:00:00", "2020-01-01 00:00:00"]
            ),
            "code": ["DX", "DX", "DX", "DX"],
            "value": ["A", "A", "B", "A"],  # categorical
            "label": [1, 1, 1, 0],
        }
    )


def test_binning_categorical_onehot_presence():
    events = _events_df()
    ds = DatasetConfig(path=Path("unused"))  # path not used by bin_events
    pp = PreprocessConfig(
        bin_size="1d",
        numeric_strategy="mean",
        categorical_strategy="onehot",
        code_selection="list",
        top_k_codes=None,
        min_code_count=1,
        code_list=["DX"],
        pipeline=[],
    )
    out = bin_events(events, ds, pp).table
    assert "cat__DX" in out.columns
    # p1 has DX in both days -> presence 1.0
    p1 = out[out["patient_id"] == "p1"].sort_values("bin_time")
    assert p1["cat__DX"].tolist() == [1.0, 1.0]


def test_binning_categorical_count():
    events = _events_df()
    ds = DatasetConfig(path=Path("unused"))
    pp = PreprocessConfig(
        bin_size="1d",
        numeric_strategy="mean",
        categorical_strategy="count",
        code_selection="list",
        top_k_codes=None,
        min_code_count=1,
        code_list=["DX"],
        pipeline=[],
    )
    out = bin_events(events, ds, pp).table
    assert "cat__DX" in out.columns
    p1 = out[out["patient_id"] == "p1"].sort_values("bin_time")
    # day1 has 2 events, day2 has 1
    assert p1["cat__DX"].tolist() == [2.0, 1.0]
