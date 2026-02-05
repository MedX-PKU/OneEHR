import pandas as pd

from oneehr.config.schema import DatasetConfig
from oneehr.datasets.tjh import load_tjh_events


def test_tjh_adapter_smoke(tmp_path):
    # Minimal wide-like TJH sample with 2 lab columns.
    df = pd.DataFrame(
        {
            "PATIENT_ID": [1.0, None, 2.0],
            "RE_DATE": pd.to_datetime(["2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-02 03:00:00"]),
            "age": [60, 60, 70],
            "gender": [1, 1, 2],
            "Admission time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "Discharge time": pd.to_datetime(["2020-01-03", "2020-01-03", "2020-01-05"]),
            "outcome": [1, 1, 0],
            "labA": [1.0, 2.0, 3.0],
            "labB": [None, 5.0, 6.0],
        }
    )
    xlsx = tmp_path / "tjh.xlsx"
    df.to_excel(xlsx, index=False)

    cfg = DatasetConfig(
        name="tjh",
        path=xlsx,
        file_type="xlsx",
        patient_id_col="patient_id",
        time_col="event_time",
        code_col="code",
        value_col="value",
        label_col="Outcome",
    )
    events = load_tjh_events(cfg)

    assert set([cfg.patient_id_col, cfg.time_col, cfg.code_col, cfg.value_col, cfg.label_col]).issubset(
        set(events.columns)
    )
    assert events[cfg.patient_id_col].nunique() == 2
    assert events[cfg.code_col].isin(["labA", "labB"]).any()
    # Sex normalized: 2 -> 0
    sex_rows = events[events[cfg.code_col] == "Sex"]
    if not sex_rows.empty:
        assert set(pd.to_numeric(sex_rows[cfg.value_col], errors="coerce").dropna().unique()).issubset({0, 1})
