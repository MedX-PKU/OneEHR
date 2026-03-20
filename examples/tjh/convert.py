#!/usr/bin/env python3
"""Convert TJH COVID-19 ICU dataset (Excel) to OneEHR input CSVs.

Reads:  examples/tjh/time_series_375_prerpocess_en.xlsx
Writes: examples/tjh/{dynamic,static}.csv
        examples/tjh/label_mortality.csv       (patient-level binary)
        examples/tjh/label_los.csv             (patient-level regression)
        examples/tjh/label_mortality_time.csv  (time-level binary)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).parent / "time_series_375_prerpocess_en.xlsx"
OUT = Path(__file__).parent

RENAME = {
    "PATIENT_ID": "PatientID",
    "outcome": "Outcome",
    "gender": "Sex",
    "age": "Age",
    "RE_DATE": "RecordTime",
    "Admission time": "AdmissionTime",
    "Discharge time": "DischargeTime",
}

LABTEST_FEATURES = [
    "Hypersensitive cardiac troponinI", "hemoglobin", "Serum chloride",
    "Prothrombin time", "procalcitonin", "eosinophils(%)",
    "Interleukin 2 receptor", "Alkaline phosphatase", "albumin",
    "basophil(%)", "Interleukin 10", "Total bilirubin", "Platelet count",
    "monocytes(%)", "antithrombin", "Interleukin 8", "indirect bilirubin",
    "Red blood cell distribution width ", "neutrophils(%)", "total protein",
    "Quantification of Treponema pallidum antibodies", "Prothrombin activity",
    "HBsAg", "mean corpuscular volume", "hematocrit",
    "White blood cell count", "Tumor necrosis factorα",
    "mean corpuscular hemoglobin concentration", "fibrinogen",
    "Interleukin 1β", "Urea", "lymphocyte count", "PH value",
    "Red blood cell count", "Eosinophil count", "Corrected calcium",
    "Serum potassium", "glucose", "neutrophils count", "Direct bilirubin",
    "Mean platelet volume", "ferritin", "RBC distribution width SD",
    "Thrombin time", "(%)lymphocyte", "HCV antibody quantification",
    "D-D dimer", "Total cholesterol", "aspartate aminotransferase",
    "Uric acid", "HCO3-", "calcium",
    "Amino-terminal brain natriuretic peptide precursor(NT-proBNP)",
    "Lactate dehydrogenase", "platelet large cell ratio ", "Interleukin 6",
    "Fibrin degradation products", "monocytes count", "PLT distribution width",
    "globulin", "γ-glutamyl transpeptidase", "International standard ratio",
    "basophil count(#)", "mean corpuscular hemoglobin ",
    "Activation of partial thromboplastin time",
    "Hypersensitive c-reactive protein", "HIV antibody quantification",
    "serum sodium", "thrombocytocrit", "ESR", "glutamic-pyruvic transaminase",
    "eGFR", "creatinine",
]


def main() -> None:
    print(f"Reading {SRC} ...")
    df = pd.read_excel(SRC)
    df = df.rename(columns=RENAME)

    # Forward-fill PatientID
    df["PatientID"] = df["PatientID"].ffill()

    # Gender: 2 → 0 (1=male, 0=female)
    df["Sex"] = df["Sex"].replace(2, 0)

    # Drop constant column
    if "2019-nCoV nucleic acid detection" in df.columns:
        df = df.drop(columns=["2019-nCoV nucleic acid detection"])

    # Truncate dates to Y-m-d
    for col in ["RecordTime", "DischargeTime", "AdmissionTime"]:
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")

    # Drop rows missing essential fields
    df = df.dropna(subset=["PatientID", "RecordTime", "DischargeTime"], how="any")

    # Set negative lab/demographic values to NaN
    num_cols = ["Sex", "Age"] + LABTEST_FEATURES
    existing = [c for c in num_cols if c in df.columns]
    df[existing] = df[existing].where(df[existing] >= 0)

    # Merge duplicate (PatientID, RecordTime) rows
    group_cols = ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"]
    df = df.groupby(group_cols, dropna=True, as_index=False).mean(numeric_only=True)

    # Ensure PatientID is int-like string
    df["PatientID"] = df["PatientID"].astype(int).astype(str)

    # --- dynamic.csv: melt labtests → long format ---
    lab_cols = [c for c in LABTEST_FEATURES if c in df.columns]
    melted = df.melt(
        id_vars=["PatientID", "RecordTime"],
        value_vars=lab_cols,
        var_name="code",
        value_name="value",
    ).dropna(subset=["value"])
    dynamic = melted.rename(columns={"PatientID": "patient_id", "RecordTime": "event_time"})
    dynamic = dynamic[["patient_id", "event_time", "code", "value"]]

    # --- static.csv: one row per patient ---
    static = (
        df.groupby("PatientID", as_index=False)
        .agg({"Age": "first", "Sex": "first"})
        .rename(columns={"PatientID": "patient_id", "Age": "age", "Sex": "gender"})
    )

    # --- Patient-level aggregates for labels ---
    patient_agg = (
        df.groupby("PatientID", as_index=False)
        .agg({
            "DischargeTime": "first",
            "AdmissionTime": "first",
            "Outcome": "first",
        })
    )
    patient_agg["PatientID"] = patient_agg["PatientID"].astype(str)

    # --- label_mortality.csv: patient-level binary (mortality) ---
    label_mortality = pd.DataFrame({
        "patient_id": patient_agg["PatientID"],
        "label_time": patient_agg["DischargeTime"],
        "label_code": "outcome",
        "label_value": patient_agg["Outcome"],
    })

    # --- label_los.csv: patient-level regression (length of stay in days) ---
    admit = pd.to_datetime(patient_agg["AdmissionTime"])
    discharge = pd.to_datetime(patient_agg["DischargeTime"])
    los_days = (discharge - admit).dt.total_seconds() / 86400.0
    label_los = pd.DataFrame({
        "patient_id": patient_agg["PatientID"],
        "label_time": patient_agg["DischargeTime"],
        "label_code": "los",
        "label_value": los_days.round(1),
    })

    # --- label_mortality_time.csv: time-level binary (mortality at each observation) ---
    # Each (patient, record_time) gets the patient's final outcome as the label.
    outcome_map = dict(zip(patient_agg["PatientID"], patient_agg["Outcome"]))
    record_times = df[["PatientID", "RecordTime"]].drop_duplicates()
    label_mortality_time = pd.DataFrame({
        "patient_id": record_times["PatientID"].values,
        "label_time": record_times["RecordTime"].values,
        "label_code": "outcome",
        "label_value": record_times["PatientID"].map(outcome_map).values,
    })

    # Write outputs
    OUT.mkdir(parents=True, exist_ok=True)
    dynamic.to_csv(OUT / "dynamic.csv", index=False)
    static.to_csv(OUT / "static.csv", index=False)
    label_mortality.to_csv(OUT / "label_mortality.csv", index=False)
    label_los.to_csv(OUT / "label_los.csv", index=False)
    label_mortality_time.to_csv(OUT / "label_mortality_time.csv", index=False)

    print(f"Wrote {len(dynamic)} dynamic rows, {len(static)} patients")
    print(f"  label_mortality.csv: {len(label_mortality)} rows (patient-level binary)")
    print(f"  label_los.csv: {len(label_los)} rows (patient-level regression)")
    print(f"  label_mortality_time.csv: {len(label_mortality_time)} rows (time-level binary)")


if __name__ == "__main__":
    main()
