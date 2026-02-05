from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.config.schema import DatasetConfig


_TJH_EXCEL_NAME = "time_series_375_prerpocess_en.xlsx"


def _resolve_tjh_excel_path(cfg: DatasetConfig) -> Path:
    if cfg.path is not None:
        return Path(cfg.path)
    if cfg.root is None:
        raise ValueError("TJH adapter requires dataset.root or dataset.path to be set.")
    return Path(cfg.root) / "raw" / _TJH_EXCEL_NAME


def _normalize_sex(series: pd.Series) -> pd.Series:
    # Match prior PyEHR convention: 1 -> male, 0 -> female (TJH uses 2 for female).
    s = series.copy()
    s = s.replace({2: 0})
    return s


def load_tjh_events(cfg: DatasetConfig) -> pd.DataFrame:
    """Load TJH raw Excel into OneEHR's normalized *event table*.

    Output schema (doctor-friendly single table):
    - patient_id_col: PatientID (string-able)
    - time_col: RecordTime (datetime)
    - code_col: feature name (string)
    - value_col: feature value (numeric or categorical)
    - label_col: label value (Outcome or LOS) repeated per patient event row

    Notes:
    - The TJH Excel stores PatientID only on the first row per patient; we forward-fill.
    - We drop the known constant column '2019-nCoV nucleic acid detection' if present.
    - We keep Admission/Discharge time columns as extra columns so users can optionally
      derive labels via label_fn or inspect dataset summaries.
    """

    path = _resolve_tjh_excel_path(cfg)
    df = pd.read_excel(path)

    # Rename to stable internal names.
    df = df.rename(
        columns={
            "PATIENT_ID": "PatientID",
            "outcome": "Outcome",
            "gender": "Sex",
            "age": "Age",
            "RE_DATE": "RecordTime",
            "Admission time": "AdmissionTime",
            "Discharge time": "DischargeTime",
        }
    )

    # Forward-fill PatientID and basic cleaning.
    if "PatientID" not in df.columns:
        raise ValueError("TJH excel missing PATIENT_ID column.")
    df["PatientID"] = df["PatientID"].ffill()

    # Ensure required columns exist.
    required = {"PatientID", "RecordTime", "AdmissionTime", "DischargeTime", "Outcome"}
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"TJH excel missing required columns: {missing}")

    # Drop rows missing key fields.
    df = df.dropna(subset=["PatientID", "RecordTime", "DischargeTime"], how="any")

    # Normalize Sex coding.
    if "Sex" in df.columns:
        df["Sex"] = _normalize_sex(df["Sex"])

    # Construct LOS label.
    los = (pd.to_datetime(df["DischargeTime"]) - pd.to_datetime(df["RecordTime"])).dt.days
    df["LOS"] = los.clip(lower=0)

    # Remove known constant / irrelevant column.
    if "2019-nCoV nucleic acid detection" in df.columns:
        df = df.drop(columns=["2019-nCoV nucleic acid detection"])

    # Aggregate to daily granularity (default, consistent with prior pipeline),
    # while preserving the original RecordTime in case users want different binning later.
    # We follow the original script: group by PatientID, RecordTime, AdmissionTime, DischargeTime and mean().
    group_cols = ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"]
    numeric_cols = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in group_cols and c not in numeric_cols]
    # For non-numeric columns, keep last after sorting by RecordTime.
    df = df.sort_values(["PatientID", "RecordTime"], kind="stable")
    df_num = df[group_cols + numeric_cols].groupby(group_cols, dropna=True, as_index=False).mean()
    if non_numeric_cols:
        df_cat = df[group_cols + non_numeric_cols].groupby(group_cols, dropna=True, as_index=False).last()
        df = df_num.merge(df_cat, on=group_cols, how="left")
    else:
        df = df_num

    # Decide which label to expose.
    label_col = cfg.label_col
    if label_col.lower() == "label":
        # Default mapping for TJH: use Outcome unless user overrides label_col.
        label_source = "Outcome"
    else:
        label_source = label_col

    if label_source not in df.columns:
        raise ValueError(
            f"TJH adapter: requested label_col={cfg.label_col!r} "
            f"(resolved to {label_source!r}) not found in TJH columns."
        )

    # Melt the wide table into event rows: code/value.
    id_cols = ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime", label_source]
    value_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="code",
        value_name="value",
    )

    # Remove missing measurements.
    long = long.dropna(subset=["value"], how="any").copy()

    # Standardize columns to DatasetConfig names.
    # Keep extra columns (Admission/Discharge) for optional label_fns / debugging.
    out = long.rename(
        columns={
            "PatientID": cfg.patient_id_col,
            "RecordTime": cfg.time_col,
            label_source: cfg.label_col,
        }
    )

    out[cfg.patient_id_col] = out[cfg.patient_id_col].astype(str)
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="raise")
    out["code"] = out["code"].astype(str)

    # Remove negative numeric values (match original preprocessing heuristic).
    # For non-numeric values, keep as-is.
    val_num = pd.to_numeric(out[cfg.value_col], errors="coerce")
    neg_mask = val_num.notna() & (val_num < 0)
    if bool(neg_mask.any()):
        out = out.loc[~neg_mask].copy()

    # Ensure label present for all rows per patient; forward-fill within patient.
    out[cfg.label_col] = out[cfg.label_col].astype(float)
    out[cfg.label_col] = out.groupby(cfg.patient_id_col, sort=False)[cfg.label_col].transform(
        lambda s: s.ffill().bfill()
    )

    return out

