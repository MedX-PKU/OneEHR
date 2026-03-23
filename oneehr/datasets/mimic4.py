"""MIMIC-IV → OneEHR converter.

Expected raw directory layout (MIMIC-IV 2.x+):
  hosp/admissions.csv, hosp/patients.csv, hosp/labevents.csv,
  hosp/diagnoses_icd.csv, hosp/procedures_icd.csv, hosp/prescriptions.csv
  icu/icustays.csv, icu/chartevents.csv

Supported label tasks:
  - mortality: in-hospital mortality per admission
  - readmission: 30-day unplanned readmission
  - los_3day / los_7day: length-of-stay binary thresholds
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.datasets._base import BaseConverter, ConvertedDataset


class MIMIC4Converter(BaseConverter):
    """Convert MIMIC-IV tables to OneEHR three-table format.

    Parameters
    ----------
    raw_dir : path
        Root directory containing ``hosp/`` and ``icu/`` subdirectories.
    use_chartevents : bool
        Whether to include ICU chart events (large). Default True.
    use_prescriptions : bool
        Whether to include prescriptions. Default True.
    max_chartevents_rows : int or None
        Limit rows read from chartevents.csv for memory control.
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
        # Detect directory layout
        self._hosp = self.raw_dir / "hosp"
        self._icu = self.raw_dir / "icu"
        if not self._hosp.is_dir():
            # Flat layout fallback
            self._hosp = self.raw_dir
            self._icu = self.raw_dir

    def _read_hosp(self, name: str, **kwargs) -> pd.DataFrame:
        path = self._hosp / name
        if not path.exists():
            gz = self._hosp / f"{name}.gz"
            if gz.exists():
                path = gz
            else:
                raise FileNotFoundError(f"Expected file not found: {path}")
        return pd.read_csv(path, **kwargs)

    def _read_icu(self, name: str, **kwargs) -> pd.DataFrame:
        path = self._icu / name
        if not path.exists():
            gz = self._icu / f"{name}.gz"
            if gz.exists():
                path = gz
            else:
                raise FileNotFoundError(f"Expected file not found: {path}")
        return pd.read_csv(path, **kwargs)

    def convert(self) -> ConvertedDataset:
        admissions = self._read_hosp("admissions.csv")
        patients = self._read_hosp("patients.csv")

        # Parse dates
        for col in ["admittime", "dischtime", "deathtime", "edregtime"]:
            if col in admissions.columns:
                admissions[col] = pd.to_datetime(admissions[col], errors="coerce")

        admissions["patient_id"] = admissions["hadm_id"].astype(str)

        # --- Dynamic table ---
        dynamic_parts: list[pd.DataFrame] = []

        # Lab events
        try:
            labs = self._read_hosp("labevents.csv")
            labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
            labs = labs.dropna(subset=["hadm_id", "charttime"])
            dynamic_parts.append(
                pd.DataFrame(
                    {
                        "patient_id": labs["hadm_id"].astype(int).astype(str),
                        "event_time": labs["charttime"],
                        "code": "LAB_" + labs["itemid"].astype(str),
                        "value": labs["value"] if "value" in labs.columns else labs["valuenum"],
                    }
                )
            )
        except FileNotFoundError:
            warnings.warn("labevents.csv not found.", stacklevel=2)

        # Chart events (ICU)
        if self.use_chartevents:
            try:
                kwargs = {}
                if self.max_chartevents_rows is not None:
                    kwargs["nrows"] = self.max_chartevents_rows
                charts = self._read_icu("chartevents.csv", **kwargs)
                charts["charttime"] = pd.to_datetime(charts["charttime"], errors="coerce")
                charts = charts.dropna(subset=["hadm_id", "charttime"])
                dynamic_parts.append(
                    pd.DataFrame(
                        {
                            "patient_id": charts["hadm_id"].astype(int).astype(str),
                            "event_time": charts["charttime"],
                            "code": "CHART_" + charts["itemid"].astype(str),
                            "value": charts["value"] if "value" in charts.columns else charts["valuenum"],
                        }
                    )
                )
            except FileNotFoundError:
                warnings.warn("chartevents.csv not found.", stacklevel=2)

        # Diagnoses (MIMIC-IV uses icd_code + icd_version)
        try:
            dx = self._read_hosp("diagnoses_icd.csv")
            dx = dx.dropna(subset=["hadm_id"])
            dx = dx.merge(
                admissions[["hadm_id", "admittime"]],
                on="hadm_id",
                how="left",
            )
            icd_version = dx["icd_version"].astype(str) if "icd_version" in dx.columns else "9"
            dynamic_parts.append(
                pd.DataFrame(
                    {
                        "patient_id": dx["hadm_id"].astype(int).astype(str),
                        "event_time": dx["admittime"],
                        "code": "DX_ICD" + icd_version + "_" + dx["icd_code"].astype(str),
                        "value": "1",
                    }
                )
            )
        except FileNotFoundError:
            warnings.warn("diagnoses_icd.csv not found.", stacklevel=2)

        # Procedures
        try:
            proc = self._read_hosp("procedures_icd.csv")
            proc = proc.dropna(subset=["hadm_id"])
            proc = proc.merge(
                admissions[["hadm_id", "admittime"]],
                on="hadm_id",
                how="left",
            )
            icd_version = proc["icd_version"].astype(str) if "icd_version" in proc.columns else "9"
            dynamic_parts.append(
                pd.DataFrame(
                    {
                        "patient_id": proc["hadm_id"].astype(int).astype(str),
                        "event_time": proc["admittime"],
                        "code": "PROC_ICD" + icd_version + "_" + proc["icd_code"].astype(str),
                        "value": "1",
                    }
                )
            )
        except FileNotFoundError:
            warnings.warn("procedures_icd.csv not found.", stacklevel=2)

        # Prescriptions
        if self.use_prescriptions:
            try:
                rx = self._read_hosp("prescriptions.csv")
                rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
                rx = rx.dropna(subset=["hadm_id", "starttime"])
                dynamic_parts.append(
                    pd.DataFrame(
                        {
                            "patient_id": rx["hadm_id"].astype(int).astype(str),
                            "event_time": rx["starttime"],
                            "code": "RX_" + rx["drug"].astype(str),
                            "value": "1",
                        }
                    )
                )
            except FileNotFoundError:
                warnings.warn("prescriptions.csv not found.", stacklevel=2)

        if not dynamic_parts:
            raise RuntimeError("No event tables found in MIMIC-IV directory.")

        dynamic = pd.concat(dynamic_parts, ignore_index=True).dropna(subset=["event_time"])

        # --- Static table ---
        adm_first = admissions.sort_values("admittime").drop_duplicates("hadm_id", keep="first")
        static_base = adm_first.merge(patients, on="subject_id", how="left")

        # MIMIC-IV stores anchor_age directly
        if "anchor_age" in static_base.columns:
            static_base["age"] = static_base["anchor_age"]
        else:
            static_base["age"] = np.nan

        static_cols = {"patient_id": static_base["hadm_id"].astype(str)}
        static_cols["age"] = static_base["age"]
        if "gender" in static_base.columns:
            static_cols["sex"] = static_base["gender"]
        if "race" in admissions.columns:
            race_map = admissions.drop_duplicates("hadm_id").set_index("hadm_id")["race"]
            static_cols["ethnicity"] = static_base["hadm_id"].map(race_map)
        elif "ethnicity" in admissions.columns:
            eth_map = admissions.drop_duplicates("hadm_id").set_index("hadm_id")["ethnicity"]
            static_cols["ethnicity"] = static_base["hadm_id"].map(eth_map)
        if "insurance" in admissions.columns:
            ins_map = admissions.drop_duplicates("hadm_id").set_index("hadm_id")["insurance"]
            static_cols["insurance"] = static_base["hadm_id"].map(ins_map)

        static = pd.DataFrame(static_cols)

        # --- Labels ---
        labels: dict[str, pd.DataFrame] = {}

        # Mortality
        mort = adm_first.copy()
        mort["died"] = mort["deathtime"].notna().astype(int)
        labels["mortality"] = pd.DataFrame(
            {
                "patient_id": mort["hadm_id"].astype(str),
                "label_time": mort["dischtime"],
                "label_code": "mortality",
                "label_value": mort["died"],
            }
        ).dropna(subset=["label_time"])

        # LOS
        adm_first["los_days"] = (adm_first["dischtime"] - adm_first["admittime"]).dt.total_seconds() / 86400
        for threshold in (3, 7):
            labels[f"los_{threshold}day"] = pd.DataFrame(
                {
                    "patient_id": adm_first["hadm_id"].astype(str),
                    "label_time": adm_first["dischtime"],
                    "label_code": f"los_{threshold}day",
                    "label_value": (adm_first["los_days"] > threshold).astype(int),
                }
            ).dropna(subset=["label_time"])

        # 30-day readmission
        adm_sorted = admissions.sort_values(["subject_id", "admittime"])
        adm_sorted["next_admit"] = adm_sorted.groupby("subject_id")["admittime"].shift(-1)
        adm_sorted["days_to_readmit"] = (adm_sorted["next_admit"] - adm_sorted["dischtime"]).dt.total_seconds() / 86400
        adm_sorted["readmit_30"] = (adm_sorted["days_to_readmit"].notna() & (adm_sorted["days_to_readmit"] <= 30)).astype(int)
        adm_sorted.loc[adm_sorted["deathtime"].notna(), "readmit_30"] = np.nan
        readmit = adm_sorted.dropna(subset=["readmit_30"])
        labels["readmission"] = pd.DataFrame(
            {
                "patient_id": readmit["hadm_id"].astype(str),
                "label_time": readmit["dischtime"],
                "label_code": "readmission",
                "label_value": readmit["readmit_30"].astype(int),
            }
        ).dropna(subset=["label_time"])

        return ConvertedDataset(dynamic=dynamic, static=static, labels=labels)
