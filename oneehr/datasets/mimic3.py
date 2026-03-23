"""MIMIC-III → OneEHR converter.

Expected raw directory layout (standard MIMIC-III CSV distribution):
  ADMISSIONS.csv, PATIENTS.csv, CHARTEVENTS.csv, LABEVENTS.csv,
  DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv, PRESCRIPTIONS.csv, ICUSTAYS.csv

Supported label tasks:
  - mortality: in-hospital mortality per ICU stay
  - readmission: 30-day unplanned readmission
  - los_3day: length-of-stay > 3 days (binary)
  - los_7day: length-of-stay > 7 days (binary)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.datasets._base import BaseConverter, ConvertedDataset


class MIMIC3Converter(BaseConverter):
    """Convert MIMIC-III tables to OneEHR three-table format.

    Parameters
    ----------
    raw_dir : path
        Directory containing the MIMIC-III CSV files.
    use_chartevents : bool
        Whether to include chart events (large file). Default True.
    use_prescriptions : bool
        Whether to include prescription events. Default True.
    max_chartevents_rows : int or None
        Limit rows read from CHARTEVENTS.csv for memory control.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        use_chartevents: bool = True,
        use_prescriptions: bool = True,
        max_chartevents_rows: int | None = None,
    ) -> None:
        super().__init__(raw_dir)
        self.use_chartevents = use_chartevents
        self.use_prescriptions = use_prescriptions
        self.max_chartevents_rows = max_chartevents_rows

    def convert(self) -> ConvertedDataset:
        # --- Load core tables ---
        admissions = self._read_csv("ADMISSIONS.csv")
        patients = self._read_csv("PATIENTS.csv")
        icustays = self._read_csv("ICUSTAYS.csv")

        # Normalize column names to uppercase
        admissions.columns = admissions.columns.str.upper()
        patients.columns = patients.columns.str.upper()
        icustays.columns = icustays.columns.str.upper()

        # Parse dates
        for col in ["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME"]:
            if col in admissions.columns:
                admissions[col] = pd.to_datetime(admissions[col], errors="coerce")
        patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")
        icustays["INTIME"] = pd.to_datetime(icustays["INTIME"], errors="coerce")
        icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"], errors="coerce")

        # Use HADM_ID as patient_id (one admission = one sample)
        # Map to string for OneEHR
        admissions["patient_id"] = admissions["HADM_ID"].astype(str)
        icustays["patient_id"] = icustays["HADM_ID"].astype(str)

        # --- Build dynamic table ---
        dynamic_parts: list[pd.DataFrame] = []

        # Lab events
        try:
            labs = self._read_csv("LABEVENTS.csv")
            labs.columns = labs.columns.str.upper()
            labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"], errors="coerce")
            labs = labs.dropna(subset=["HADM_ID", "CHARTTIME"])
            labs_dynamic = pd.DataFrame(
                {
                    "patient_id": labs["HADM_ID"].astype(int).astype(str),
                    "event_time": labs["CHARTTIME"],
                    "code": "LAB_" + labs["ITEMID"].astype(str),
                    "value": labs["VALUE"],
                }
            )
            dynamic_parts.append(labs_dynamic)
        except FileNotFoundError:
            warnings.warn("LABEVENTS.csv not found, skipping lab events.", stacklevel=2)

        # Chart events
        if self.use_chartevents:
            try:
                kwargs = {}
                if self.max_chartevents_rows is not None:
                    kwargs["nrows"] = self.max_chartevents_rows
                charts = self._read_csv("CHARTEVENTS.csv", **kwargs)
                charts.columns = charts.columns.str.upper()
                charts["CHARTTIME"] = pd.to_datetime(charts["CHARTTIME"], errors="coerce")
                charts = charts.dropna(subset=["HADM_ID", "CHARTTIME"])
                charts_dynamic = pd.DataFrame(
                    {
                        "patient_id": charts["HADM_ID"].astype(int).astype(str),
                        "event_time": charts["CHARTTIME"],
                        "code": "CHART_" + charts["ITEMID"].astype(str),
                        "value": charts["VALUE"],
                    }
                )
                dynamic_parts.append(charts_dynamic)
            except FileNotFoundError:
                warnings.warn("CHARTEVENTS.csv not found, skipping chart events.", stacklevel=2)

        # Diagnoses
        try:
            dx = self._read_csv("DIAGNOSES_ICD.csv")
            dx.columns = dx.columns.str.upper()
            dx = dx.dropna(subset=["HADM_ID"])
            # Use admission time as event time for diagnoses
            dx = dx.merge(
                admissions[["HADM_ID", "ADMITTIME"]],
                on="HADM_ID",
                how="left",
            )
            dx_dynamic = pd.DataFrame(
                {
                    "patient_id": dx["HADM_ID"].astype(int).astype(str),
                    "event_time": dx["ADMITTIME"],
                    "code": "DX_" + dx["ICD9_CODE"].astype(str),
                    "value": "1",
                }
            )
            dynamic_parts.append(dx_dynamic)
        except FileNotFoundError:
            warnings.warn("DIAGNOSES_ICD.csv not found.", stacklevel=2)

        # Procedures
        try:
            proc = self._read_csv("PROCEDURES_ICD.csv")
            proc.columns = proc.columns.str.upper()
            proc = proc.dropna(subset=["HADM_ID"])
            proc = proc.merge(
                admissions[["HADM_ID", "ADMITTIME"]],
                on="HADM_ID",
                how="left",
            )
            proc_dynamic = pd.DataFrame(
                {
                    "patient_id": proc["HADM_ID"].astype(int).astype(str),
                    "event_time": proc["ADMITTIME"],
                    "code": "PROC_" + proc["ICD9_CODE"].astype(str),
                    "value": "1",
                }
            )
            dynamic_parts.append(proc_dynamic)
        except FileNotFoundError:
            warnings.warn("PROCEDURES_ICD.csv not found.", stacklevel=2)

        # Prescriptions
        if self.use_prescriptions:
            try:
                rx = self._read_csv("PRESCRIPTIONS.csv")
                rx.columns = rx.columns.str.upper()
                rx["STARTDATE"] = pd.to_datetime(rx["STARTDATE"], errors="coerce")
                rx = rx.dropna(subset=["HADM_ID", "STARTDATE"])
                drug_col = "DRUG" if "DRUG" in rx.columns else "FORMULARY_DRUG_CD"
                rx_dynamic = pd.DataFrame(
                    {
                        "patient_id": rx["HADM_ID"].astype(int).astype(str),
                        "event_time": rx["STARTDATE"],
                        "code": "RX_" + rx[drug_col].astype(str),
                        "value": "1",
                    }
                )
                dynamic_parts.append(rx_dynamic)
            except FileNotFoundError:
                warnings.warn("PRESCRIPTIONS.csv not found.", stacklevel=2)

        if not dynamic_parts:
            raise RuntimeError("No event tables found in MIMIC-III directory.")

        dynamic = pd.concat(dynamic_parts, ignore_index=True)
        dynamic = dynamic.dropna(subset=["event_time"])

        # --- Build static table ---
        # Merge patient demographics with first admission info
        adm_first = admissions.sort_values("ADMITTIME").drop_duplicates("HADM_ID", keep="first")
        static_base = adm_first.merge(patients, on="SUBJECT_ID", how="left")

        # Compute age at admission
        static_base["age"] = ((static_base["ADMITTIME"] - static_base["DOB"]).dt.days / 365.25).round(1)
        # Cap unrealistic ages (MIMIC-III shifts DOB for >89)
        static_base.loc[static_base["age"] > 200, "age"] = 91.4

        sex_col = "GENDER" if "GENDER" in static_base.columns else "SEX"
        ethnicity_col = "ETHNICITY" if "ETHNICITY" in static_base.columns else None

        static_cols = {"patient_id": static_base["HADM_ID"].astype(str), "age": static_base["age"]}
        if sex_col in static_base.columns:
            static_cols["sex"] = static_base[sex_col]
        if ethnicity_col and ethnicity_col in static_base.columns:
            static_cols["ethnicity"] = static_base[ethnicity_col]
        if "INSURANCE" in static_base.columns:
            static_cols["insurance"] = static_base["INSURANCE"]

        static = pd.DataFrame(static_cols)

        # --- Build label tables ---
        labels: dict[str, pd.DataFrame] = {}

        # In-hospital mortality
        mort = adm_first.copy()
        mort["died"] = mort["DEATHTIME"].notna().astype(int)
        labels["mortality"] = pd.DataFrame(
            {
                "patient_id": mort["HADM_ID"].astype(str),
                "label_time": mort["DISCHTIME"],
                "label_code": "mortality",
                "label_value": mort["died"],
            }
        ).dropna(subset=["label_time"])

        # Length of stay
        adm_first["los_days"] = (adm_first["DISCHTIME"] - adm_first["ADMITTIME"]).dt.total_seconds() / 86400
        labels["los_3day"] = pd.DataFrame(
            {
                "patient_id": adm_first["HADM_ID"].astype(str),
                "label_time": adm_first["DISCHTIME"],
                "label_code": "los_3day",
                "label_value": (adm_first["los_days"] > 3).astype(int),
            }
        ).dropna(subset=["label_time"])

        labels["los_7day"] = pd.DataFrame(
            {
                "patient_id": adm_first["HADM_ID"].astype(str),
                "label_time": adm_first["DISCHTIME"],
                "label_code": "los_7day",
                "label_value": (adm_first["los_days"] > 7).astype(int),
            }
        ).dropna(subset=["label_time"])

        # 30-day readmission
        adm_sorted = admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])
        adm_sorted["next_admit"] = adm_sorted.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
        adm_sorted["days_to_readmit"] = (adm_sorted["next_admit"] - adm_sorted["DISCHTIME"]).dt.total_seconds() / 86400
        adm_sorted["readmit_30"] = (adm_sorted["days_to_readmit"].notna() & (adm_sorted["days_to_readmit"] <= 30)).astype(int)
        # Exclude patients who died (can't be readmitted)
        adm_sorted.loc[adm_sorted["DEATHTIME"].notna(), "readmit_30"] = np.nan

        readmit = adm_sorted.dropna(subset=["readmit_30"])
        labels["readmission"] = pd.DataFrame(
            {
                "patient_id": readmit["HADM_ID"].astype(str),
                "label_time": readmit["DISCHTIME"],
                "label_code": "readmission",
                "label_value": readmit["readmit_30"].astype(int),
            }
        ).dropna(subset=["label_time"])

        return ConvertedDataset(dynamic=dynamic, static=static, labels=labels)
