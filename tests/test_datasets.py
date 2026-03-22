"""Tests for dataset converters (using synthetic data)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_mimic3_dir(tmp_path: Path) -> Path:
    """Create a minimal MIMIC-III-like directory with synthetic data."""
    raw = tmp_path / "mimic3"
    raw.mkdir()

    # PATIENTS.csv
    pd.DataFrame({
        "SUBJECT_ID": [1, 2, 3],
        "GENDER": ["M", "F", "M"],
        "DOB": ["1950-01-01", "1960-06-15", "1970-03-20"],
    }).to_csv(raw / "PATIENTS.csv", index=False)

    # ADMISSIONS.csv
    pd.DataFrame({
        "HADM_ID": [100, 200, 300],
        "SUBJECT_ID": [1, 2, 3],
        "ADMITTIME": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "DISCHTIME": ["2020-01-05", "2020-02-03", "2020-03-10"],
        "DEATHTIME": [None, None, "2020-03-10"],
        "ETHNICITY": ["WHITE", "BLACK", "ASIAN"],
        "INSURANCE": ["Medicare", "Private", "Medicaid"],
    }).to_csv(raw / "ADMISSIONS.csv", index=False)

    # ICUSTAYS.csv
    pd.DataFrame({
        "ICUSTAY_ID": [1001, 2001, 3001],
        "HADM_ID": [100, 200, 300],
        "SUBJECT_ID": [1, 2, 3],
        "INTIME": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "OUTTIME": ["2020-01-05", "2020-02-03", "2020-03-10"],
    }).to_csv(raw / "ICUSTAYS.csv", index=False)

    # LABEVENTS.csv
    rng = np.random.default_rng(42)
    lab_rows = []
    for hadm in [100, 200, 300]:
        for day in range(3):
            lab_rows.append({
                "HADM_ID": hadm,
                "ITEMID": 50801,
                "CHARTTIME": f"2020-0{hadm // 100}-0{day + 1}",
                "VALUE": str(round(rng.normal(7.4, 0.1), 2)),
            })
            lab_rows.append({
                "HADM_ID": hadm,
                "ITEMID": 50802,
                "CHARTTIME": f"2020-0{hadm // 100}-0{day + 1}",
                "VALUE": str(round(rng.normal(120, 20), 1)),
            })
    pd.DataFrame(lab_rows).to_csv(raw / "LABEVENTS.csv", index=False)

    # DIAGNOSES_ICD.csv
    pd.DataFrame({
        "HADM_ID": [100, 100, 200, 300],
        "ICD9_CODE": ["4019", "25000", "4019", "5849"],
        "SEQ_NUM": [1, 2, 1, 1],
    }).to_csv(raw / "DIAGNOSES_ICD.csv", index=False)

    return raw


def test_mimic3_converter(tmp_path):
    from oneehr.datasets.mimic3 import MIMIC3Converter

    raw = _make_mimic3_dir(tmp_path)
    converter = MIMIC3Converter(raw, use_chartevents=False, use_prescriptions=False)
    result = converter.convert()

    assert not result.dynamic.empty
    assert "patient_id" in result.dynamic.columns
    assert "event_time" in result.dynamic.columns
    assert "code" in result.dynamic.columns
    assert "value" in result.dynamic.columns

    assert not result.static.empty
    assert "patient_id" in result.static.columns
    assert "age" in result.static.columns

    assert "mortality" in result.labels
    assert "readmission" in result.labels
    assert "los_3day" in result.labels


def test_mimic3_converter_save(tmp_path):
    from oneehr.datasets.mimic3 import MIMIC3Converter

    raw = _make_mimic3_dir(tmp_path)
    out = tmp_path / "output"
    converter = MIMIC3Converter(raw, use_chartevents=False, use_prescriptions=False)
    paths = converter.save(out, task="mortality")

    assert "dynamic" in paths
    assert "static" in paths
    assert "label" in paths
    assert paths["dynamic"].exists()
    assert paths["static"].exists()
    assert paths["label"].exists()


def _make_eicu_dir(tmp_path: Path) -> Path:
    """Create a minimal eICU-like directory."""
    raw = tmp_path / "eicu"
    raw.mkdir()

    pd.DataFrame({
        "patientunitstayid": [1, 2, 3],
        "age": ["65", "72", "> 89"],
        "gender": ["Male", "Female", "Male"],
        "ethnicity": ["Caucasian", "African American", "Asian"],
        "hospitaladmitoffset": [-120, -60, -180],
        "unitdischargeoffset": [4320, 1440, 7200],
        "unitdischargestatus": ["Alive", "Alive", "Expired"],
    }).to_csv(raw / "patient.csv", index=False)

    # lab.csv
    lab_rows = []
    for pid in [1, 2, 3]:
        for offset in [0, 60, 120]:
            lab_rows.append({
                "patientunitstayid": pid,
                "labresultoffset": offset,
                "labname": "glucose",
                "labresult": round(np.random.default_rng(42).normal(100, 20), 1),
            })
    pd.DataFrame(lab_rows).to_csv(raw / "lab.csv", index=False)

    return raw


def test_eicu_converter(tmp_path):
    from oneehr.datasets.eicu import EICUConverter

    raw = _make_eicu_dir(tmp_path)
    converter = EICUConverter(raw, use_vitals=False, use_medication=False)
    result = converter.convert()

    assert not result.dynamic.empty
    assert not result.static.empty
    assert "mortality" in result.labels
    assert "los_3day" in result.labels


def test_converter_missing_dir(tmp_path):
    from oneehr.datasets.mimic3 import MIMIC3Converter

    with pytest.raises(FileNotFoundError):
        MIMIC3Converter(tmp_path / "nonexistent")
