import pandas as pd

from oneehr.config.schema import DatasetConfig
from oneehr.data.converters.tjh import convert


def test_tjh_converter_outputs_unified_schema():
    df_raw = pd.DataFrame(
        {
            "PATIENT_ID": [1.0, None, 2.0],
            "RE_DATE": pd.to_datetime(
                ["2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-02 03:00:00"]
            ),
            "age": [60, 60, 70],
            "gender": [1, 1, 2],
            "Admission time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "Discharge time": pd.to_datetime(["2020-01-03", "2020-01-03", "2020-01-05"]),
            "outcome": [1, 1, 0],
            "labA": [1.0, 2.0, 3.0],
            "labB": [None, 5.0, 6.0],
        }
    )
    cfg = DatasetConfig(
        path=__file__,
        file_type="xlsx",
        patient_id_col="patient_id",
        time_col="event_time",
        code_col="code",
        value_col="value",
    )
    res = convert(df_raw, cfg)
    events = res.events
    assert set(["patient_id", "event_time", "code", "value"]).issubset(set(events.columns))
    assert events["patient_id"].nunique() == 2
    assert events["code"].isin(["labA", "labB"]).any()


def test_tjh_builtin_label_fns(tmp_path):
    df_raw = pd.DataFrame(
        {
            "PATIENT_ID": [1.0, None],
            "RE_DATE": pd.to_datetime(["2020-01-01 01:00:00", "2020-01-01 02:00:00"]),
            "age": [60, 60],
            "gender": [1, 1],
            "Admission time": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "Discharge time": pd.to_datetime(["2020-01-03", "2020-01-03"]),
            "outcome": [1, 1],
            "labA": [1.0, 2.0],
        }
    )
    cfg = DatasetConfig(
        path=__file__,
        file_type="xlsx",
        patient_id_col="patient_id",
        time_col="event_time",
        code_col="code",
        value_col="value",
    )
    events = convert(df_raw, cfg).events
    from types import SimpleNamespace

    from oneehr.data.converters.tjh import build_labels

    cfg_out = SimpleNamespace(dataset=SimpleNamespace(patient_id_col="patient_id", time_col="event_time"), task=SimpleNamespace(kind="binary"))
    y_out = build_labels(events, cfg_out)
    assert set(y_out.columns) == {"patient_id", "label"}
    assert int(y_out["label"].iloc[0]) == 1

    cfg_los = SimpleNamespace(dataset=SimpleNamespace(patient_id_col="patient_id", time_col="event_time"), task=SimpleNamespace(kind="regression"))
    y_los = build_labels(events, cfg_los)
    assert set(y_los.columns) == {"patient_id", "label"}
    assert int(y_los["label"].iloc[0]) == 2
