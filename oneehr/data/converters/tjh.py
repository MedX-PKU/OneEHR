from __future__ import annotations

import pandas as pd

from dataclasses import dataclass

from oneehr.config.schema import DatasetConfig
from oneehr.data.converters.schema import ConvertedDataset


@dataclass(frozen=True)
class TJHConverted:
    events: pd.DataFrame
    labels: dict[str, pd.DataFrame]


def convert(df_raw: pd.DataFrame, cfg: DatasetConfig) -> TJHConverted:
    """Convert TJH raw Excel (wide) to OneEHR unified event table (long).

    Output columns (minimum):
    - patient_id (cfg.patient_id_col)
    - event_time (cfg.time_col)
    - code (cfg.code_col)
    - value (cfg.value_col)

    In addition to `events`, this converter also prepares common labels for TJH
    tasks (single-task per run):
    - outcome (binary): columns [patient_id, label]
    - los (regression): columns [patient_id, label]
    """

    df = df_raw.rename(
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

    if "PatientID" not in df.columns:
        raise ValueError("TJH raw file missing PATIENT_ID column.")
    if "RecordTime" not in df.columns:
        raise ValueError("TJH raw file missing RE_DATE column.")

    # TJH only records PatientID on first row of each patient.
    df["PatientID"] = df["PatientID"].ffill()
    df = df.dropna(subset=["PatientID", "RecordTime"], how="any").copy()

    # Normalize Sex: 1 male, 2 female -> 0.
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].replace({2: 0})

    # Drop known constant column.
    if "2019-nCoV nucleic acid detection" in df.columns:
        df = df.drop(columns=["2019-nCoV nucleic acid detection"])

    # Daily merge as legacy preprocessing: group by patient + record time (+ admission/discharge if present).
    group_cols = [c for c in ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"] if c in df.columns]
    df = df.sort_values(["PatientID", "RecordTime"], kind="stable")
    numeric_cols = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in group_cols and c not in numeric_cols]
    df_num = df[group_cols + numeric_cols].groupby(group_cols, dropna=True, as_index=False).mean()
    if non_numeric_cols:
        df_cat = df[group_cols + non_numeric_cols].groupby(group_cols, dropna=True, as_index=False).last()
        df = df_num.merge(df_cat, on=group_cols, how="left")
    else:
        df = df_num

    # Exclude target columns from codes; label_fn will use them as metadata columns.
    target_like = {"Outcome", "LOS"}
    value_cols = [c for c in df.columns if c not in set(group_cols) and c not in target_like]

    long = df.melt(
        id_vars=group_cols,
        value_vars=value_cols,
        var_name=cfg.code_col,
        value_name=cfg.value_col,
    )
    long = long.dropna(subset=[cfg.value_col], how="any").copy()

    out = long.rename(
        columns={
            "PatientID": cfg.patient_id_col,
            "RecordTime": cfg.time_col,
        }
    )
    out[cfg.patient_id_col] = out[cfg.patient_id_col].astype(str)
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="raise")
    out[cfg.code_col] = out[cfg.code_col].astype(str)

    # Attach useful metadata columns (repeated per event row) for label_fn/static_features.
    meta_cols = [c for c in ["AdmissionTime", "DischargeTime", "Sex", "Age", "Outcome"] if c in df.columns]
    if meta_cols:
        meta = df[["PatientID", "RecordTime", *meta_cols]].copy()
        meta = meta.rename(columns={"PatientID": cfg.patient_id_col, "RecordTime": cfg.time_col})
        meta[cfg.patient_id_col] = meta[cfg.patient_id_col].astype(str)
        meta[cfg.time_col] = pd.to_datetime(meta[cfg.time_col], errors="raise")
        # Ensure no duplicate columns in out before merge (avoid suffixes).
        dup = [c for c in meta_cols if c in out.columns]
        if dup:
            out = out.drop(columns=dup)
        out = out.merge(meta, on=[cfg.patient_id_col, cfg.time_col], how="left")

    # Drop negative numeric measurements.
    val_num = pd.to_numeric(out[cfg.value_col], errors="coerce")
    neg = val_num.notna() & (val_num < 0)
    if bool(neg.any()):
        out = out.loc[~neg].copy()

    # Built-in labels (patient-level)
    # Outcome: one label per patient
    labels: dict[str, pd.DataFrame] = {}
    if "Outcome" in out.columns:
        y = out[[cfg.patient_id_col, "Outcome"]].dropna(subset=["Outcome"]).drop_duplicates(
            subset=[cfg.patient_id_col],
            keep="last",
        )
        labels["outcome"] = y.rename(columns={"Outcome": "label"})[[cfg.patient_id_col, "label"]].copy()

    # LOS: use date-level discharge - record date (match legacy behavior).
    if "DischargeTime" in out.columns:
        base = (
            out[[cfg.patient_id_col, cfg.time_col, "DischargeTime"]]
            .dropna(subset=[cfg.time_col, "DischargeTime"])
            .sort_values([cfg.patient_id_col, cfg.time_col], kind="stable")
            .groupby(cfg.patient_id_col, sort=False)
            .first()
        )
        los = (pd.to_datetime(base["DischargeTime"]).dt.normalize() - pd.to_datetime(base[cfg.time_col]).dt.normalize()).dt.days
        los = los.clip(lower=0)
        labels["los"] = los.rename("label").reset_index()[[cfg.patient_id_col, "label"]]

    return TJHConverted(events=out, labels=labels)


def convert_events(df_raw: pd.DataFrame, cfg: DatasetConfig) -> ConvertedDataset:
    """Compatibility helper returning the generic ConvertedDataset.

    Recommended for external converters: return `ConvertedDataset(events=...)`.
    """

    res = convert(df_raw, cfg)
    return ConvertedDataset(events=res.events, meta={"labels_keys": sorted(res.labels.keys())})


def build_labels(events: pd.DataFrame, cfg) -> pd.DataFrame:
    """Select a label table from TJHConverted.labels.

    Intended to be used as `labels.fn = "oneehr/data/converters/tjh.py:build_labels"`.
    Uses cfg.task.kind to choose:
    - binary -> outcome
    - regression -> los
    """

    # Note: `events` here is the converted event table (with Outcome/DischargeTime metadata).
    # We recompute using the same logic to avoid having to thread labels through RunIO.
    pid_col = cfg.dataset.patient_id_col
    time_col = cfg.dataset.time_col

    if cfg.task.kind == "binary":
        if "Outcome" not in events.columns:
            raise ValueError("TJH build_labels(binary) requires Outcome column on events")
        y = events[[pid_col, "Outcome"]].dropna(subset=["Outcome"]).drop_duplicates(
            subset=[pid_col],
            keep="last",
        )
        out = y.rename(columns={"Outcome": "label"})[[pid_col, "label"]].copy()
        out = out.rename(columns={pid_col: "patient_id"})
        return out

    if cfg.task.kind == "regression":
        if "DischargeTime" not in events.columns:
            raise ValueError("TJH build_labels(regression) requires DischargeTime column on events")
        df = events[[pid_col, time_col, "DischargeTime"]].dropna(subset=[time_col, "DischargeTime"]).copy()
        df[pid_col] = df[pid_col].astype(str)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["DischargeTime"] = pd.to_datetime(df["DischargeTime"], errors="coerce")
        df = df.dropna(subset=[time_col, "DischargeTime"])
        base = (
            df.sort_values([pid_col, time_col], kind="stable")
            .groupby(pid_col, sort=False)
            .first()
        )
        los = (base["DischargeTime"].dt.normalize() - base[time_col].dt.normalize()).dt.days.clip(lower=0)
        out = los.rename("label").reset_index().rename(columns={pid_col: "patient_id"})
        return out[["patient_id", "label"]]

    raise ValueError(f"Unsupported task.kind for TJH build_labels: {cfg.task.kind!r}")
