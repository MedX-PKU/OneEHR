"""eICU Collaborative Research Database → OneEHR converter.

Expected raw directory layout:
  patient.csv, lab.csv, vitalPeriodic.csv, vitalAperiodic.csv,
  diagnosis.csv, medication.csv, admissionDx.csv

Supported label tasks:
  - mortality: in-hospital mortality
  - los_3day / los_7day: length-of-stay binary thresholds
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from oneehr.datasets._base import BaseConverter, ConvertedDataset


class EICUConverter(BaseConverter):
    """Convert eICU tables to OneEHR three-table format.

    Parameters
    ----------
    raw_dir : path
        Directory containing the eICU CSV files.
    use_vitals : bool
        Whether to include vital sign events. Default True.
    use_medication : bool
        Whether to include medication events. Default True.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        use_vitals: bool = True,
        use_medication: bool = True,
    ) -> None:
        super().__init__(raw_dir)
        self.use_vitals = use_vitals
        self.use_medication = use_medication

    def convert(self) -> ConvertedDataset:
        patient = self._read_csv("patient.csv")

        # eICU uses patientunitstayid as unique ICU stay identifier
        patient["patient_id"] = patient["patientunitstayid"].astype(str)

        # Compute reference time (hospital admission)
        # eICU stores offsets in minutes from hospital admit
        # We create a synthetic base timestamp per stay
        patient["base_time"] = pd.Timestamp("2020-01-01")
        patient["admit_time"] = patient["base_time"] + pd.to_timedelta(
            patient.get("hospitaladmitoffset", 0).fillna(0).clip(upper=0).abs(),
            unit="min",
        )

        # --- Dynamic table ---
        dynamic_parts: list[pd.DataFrame] = []

        # Lab events
        try:
            lab = self._read_csv("lab.csv")
            lab = lab.merge(
                patient[["patientunitstayid", "base_time"]],
                on="patientunitstayid",
                how="left",
            )
            lab["event_time"] = lab["base_time"] + pd.to_timedelta(lab["labresultoffset"].fillna(0), unit="min")
            dynamic_parts.append(
                pd.DataFrame(
                    {
                        "patient_id": lab["patientunitstayid"].astype(str),
                        "event_time": lab["event_time"],
                        "code": "LAB_" + lab["labname"].astype(str),
                        "value": lab["labresult"],
                    }
                )
            )
        except FileNotFoundError:
            warnings.warn("lab.csv not found.", stacklevel=2)

        # Vital signs (periodic)
        if self.use_vitals:
            try:
                vitals = self._read_csv("vitalPeriodic.csv")
                vitals = vitals.merge(
                    patient[["patientunitstayid", "base_time"]],
                    on="patientunitstayid",
                    how="left",
                )
                vitals["event_time"] = vitals["base_time"] + pd.to_timedelta(vitals["observationoffset"].fillna(0), unit="min")
                # Melt vital columns into long format
                vital_cols = [
                    c
                    for c in vitals.columns
                    if c
                    not in {
                        "patientunitstayid",
                        "vitalperiodicid",
                        "observationoffset",
                        "base_time",
                        "event_time",
                    }
                ]
                for col in vital_cols:
                    part = vitals[["patientunitstayid", "event_time", col]].dropna(subset=[col])
                    if part.empty:
                        continue
                    dynamic_parts.append(
                        pd.DataFrame(
                            {
                                "patient_id": part["patientunitstayid"].astype(str),
                                "event_time": part["event_time"],
                                "code": f"VITAL_{col}",
                                "value": part[col],
                            }
                        )
                    )
            except FileNotFoundError:
                warnings.warn("vitalPeriodic.csv not found.", stacklevel=2)

            # Aperiodic vitals
            try:
                avitals = self._read_csv("vitalAperiodic.csv")
                avitals = avitals.merge(
                    patient[["patientunitstayid", "base_time"]],
                    on="patientunitstayid",
                    how="left",
                )
                avitals["event_time"] = avitals["base_time"] + pd.to_timedelta(avitals["observationoffset"].fillna(0), unit="min")
                avital_cols = [
                    c
                    for c in avitals.columns
                    if c
                    not in {
                        "patientunitstayid",
                        "vitalaperiodicid",
                        "observationoffset",
                        "base_time",
                        "event_time",
                    }
                ]
                for col in avital_cols:
                    part = avitals[["patientunitstayid", "event_time", col]].dropna(subset=[col])
                    if part.empty:
                        continue
                    dynamic_parts.append(
                        pd.DataFrame(
                            {
                                "patient_id": part["patientunitstayid"].astype(str),
                                "event_time": part["event_time"],
                                "code": f"VITAL_{col}",
                                "value": part[col],
                            }
                        )
                    )
            except FileNotFoundError:
                warnings.warn("vitalAperiodic.csv not found.", stacklevel=2)

        # Diagnoses
        try:
            dx = self._read_csv("diagnosis.csv")
            dx = dx.merge(
                patient[["patientunitstayid", "base_time"]],
                on="patientunitstayid",
                how="left",
            )
            dx["event_time"] = dx["base_time"] + pd.to_timedelta(dx["diagnosisoffset"].fillna(0), unit="min")
            code_col = "icd9code" if "icd9code" in dx.columns else "diagnosisstring"
            dynamic_parts.append(
                pd.DataFrame(
                    {
                        "patient_id": dx["patientunitstayid"].astype(str),
                        "event_time": dx["event_time"],
                        "code": "DX_" + dx[code_col].astype(str),
                        "value": "1",
                    }
                )
            )
        except FileNotFoundError:
            warnings.warn("diagnosis.csv not found.", stacklevel=2)

        # Medications
        if self.use_medication:
            try:
                med = self._read_csv("medication.csv")
                med = med.merge(
                    patient[["patientunitstayid", "base_time"]],
                    on="patientunitstayid",
                    how="left",
                )
                offset_col = "drugstartoffset" if "drugstartoffset" in med.columns else "drugorderoffset"
                med["event_time"] = med["base_time"] + pd.to_timedelta(med[offset_col].fillna(0), unit="min")
                drug_col = "drugname" if "drugname" in med.columns else "drughiclseqno"
                dynamic_parts.append(
                    pd.DataFrame(
                        {
                            "patient_id": med["patientunitstayid"].astype(str),
                            "event_time": med["event_time"],
                            "code": "RX_" + med[drug_col].astype(str),
                            "value": "1",
                        }
                    )
                )
            except FileNotFoundError:
                warnings.warn("medication.csv not found.", stacklevel=2)

        if not dynamic_parts:
            raise RuntimeError("No event tables found in eICU directory.")

        dynamic = pd.concat(dynamic_parts, ignore_index=True).dropna(subset=["event_time"])

        # --- Static table ---
        static_cols = {"patient_id": patient["patient_id"]}
        if "age" in patient.columns:
            # eICU stores age as string "> 89" for elderly
            age = pd.to_numeric(patient["age"], errors="coerce")
            age = age.fillna(91.4)  # convention for >89
            static_cols["age"] = age
        if "gender" in patient.columns:
            static_cols["sex"] = patient["gender"]
        if "ethnicity" in patient.columns:
            static_cols["ethnicity"] = patient["ethnicity"]

        static = pd.DataFrame(static_cols)

        # --- Labels ---
        labels: dict[str, pd.DataFrame] = {}

        # Mortality
        if "unitdischargestatus" in patient.columns:
            died = (patient["unitdischargestatus"] == "Expired").astype(int)
        elif "hospitaldischargestatus" in patient.columns:
            died = (patient["hospitaldischargestatus"] == "Expired").astype(int)
        else:
            died = pd.Series(0, index=patient.index)

        # Discharge time
        if "unitdischargeoffset" in patient.columns:
            disch_time = patient["base_time"] + pd.to_timedelta(patient["unitdischargeoffset"].fillna(0), unit="min")
        else:
            disch_time = patient["base_time"] + pd.Timedelta(days=1)

        labels["mortality"] = pd.DataFrame(
            {
                "patient_id": patient["patient_id"],
                "label_time": disch_time,
                "label_code": "mortality",
                "label_value": died,
            }
        )

        # LOS
        if "unitdischargeoffset" in patient.columns:
            los_days = patient["unitdischargeoffset"].fillna(0) / (60 * 24)
            for threshold in (3, 7):
                labels[f"los_{threshold}day"] = pd.DataFrame(
                    {
                        "patient_id": patient["patient_id"],
                        "label_time": disch_time,
                        "label_code": f"los_{threshold}day",
                        "label_value": (los_days > threshold).astype(int),
                    }
                )

        return ConvertedDataset(dynamic=dynamic, static=static, labels=labels)
