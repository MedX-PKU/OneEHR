import pandas as pd

from oneehr.artifacts.labels import validate_patient_labels, validate_time_labels


def test_validate_patient_labels():
    df = pd.DataFrame({"patient_id": [1, 2], "label": [0.0, 1.0], "extra": [9, 9]})
    out = validate_patient_labels(df)
    assert list(out.columns) == ["patient_id", "label"]
    assert pd.api.types.is_string_dtype(out["patient_id"].dtype)


def test_validate_time_labels():
    df = pd.DataFrame(
        {"patient_id": [1], "bin_time": pd.to_datetime(["2020-01-01"]), "label": [1.0], "mask": [True], "x": [0]}
    )
    out = validate_time_labels(df)
    assert list(out.columns) == ["patient_id", "bin_time", "label", "mask"]
    assert pd.api.types.is_string_dtype(out["patient_id"].dtype)
