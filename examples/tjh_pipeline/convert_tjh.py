from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"TJH raw must be an Excel file. Got: {path}")
    try:
        return pd.read_excel(path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "Reading TJH Excel inputs requires openpyxl. "
            "Install dependencies with `uv pip install -e .` or `uv pip install openpyxl`."
        ) from exc


def _normalize_tjh(df_raw: pd.DataFrame) -> pd.DataFrame:
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

    return df


def build_dynamic(df_norm: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"] if c in df_norm.columns]
    # Keep Outcome/LOS-related fields out of dynamic codes; they go to label.csv instead.
    label_like = {"Outcome", "LOS"}
    value_cols = [c for c in df_norm.columns if c not in set(group_cols) and c not in label_like]

    long = df_norm.melt(
        id_vars=group_cols,
        value_vars=value_cols,
        var_name="code",
        value_name="value",
    ).dropna(subset=["value"])

    out = long.rename(columns={"PatientID": "patient_id", "RecordTime": "event_time"}).copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["event_time"] = pd.to_datetime(out["event_time"], errors="raise")
    out["code"] = out["code"].astype(str)
    return out[["patient_id", "event_time", "code", "value"]]


def build_static(df_norm: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["PatientID", "Sex", "Age"] if c in df_norm.columns]
    if "PatientID" not in cols:
        raise ValueError("Missing PatientID in normalized TJH dataframe.")
    if len(cols) == 1:
        # Only patient_id exists; still create a valid static.csv.
        out = df_norm[["PatientID"]].drop_duplicates().copy()
    else:
        # Take first observed row per patient for static fields.
        out = (
            df_norm[cols]
            .sort_values(["PatientID"], kind="stable")
            .drop_duplicates(subset=["PatientID"], keep="first")
            .copy()
        )
    out = out.rename(columns={"PatientID": "patient_id"})
    out["patient_id"] = out["patient_id"].astype(str)
    return out


def build_label(df_norm: pd.DataFrame) -> pd.DataFrame:
    # Outcome: one label per patient (last).
    out_frames: list[pd.DataFrame] = []
    if "Outcome" in df_norm.columns and "PatientID" in df_norm.columns and "RecordTime" in df_norm.columns:
        y = (
            df_norm[["PatientID", "RecordTime", "Outcome"]]
            .dropna(subset=["Outcome"])
            .sort_values(["PatientID", "RecordTime"], kind="stable")
            .groupby("PatientID", sort=False)
            .tail(1)
        )
        y = y.rename(columns={"PatientID": "patient_id", "RecordTime": "label_time", "Outcome": "label_value"})
        y["label_code"] = "outcome"
        out_frames.append(y[["patient_id", "label_time", "label_code", "label_value"]])

    # LOS: discharge date - first record date per patient.
    if "DischargeTime" in df_norm.columns and "PatientID" in df_norm.columns and "RecordTime" in df_norm.columns:
        base = (
            df_norm[["PatientID", "RecordTime", "DischargeTime"]]
            .dropna(subset=["RecordTime", "DischargeTime"])
            .sort_values(["PatientID", "RecordTime"], kind="stable")
            .groupby("PatientID", sort=False)
            .first()
        )
        los = (pd.to_datetime(base["DischargeTime"]).dt.normalize() - pd.to_datetime(base["RecordTime"]).dt.normalize()).dt.days
        los = los.clip(lower=0)
        y = los.rename("label_value").reset_index().rename(columns={"PatientID": "patient_id"})
        y["label_time"] = pd.to_datetime(base["RecordTime"]).values
        y["label_code"] = "los"
        out_frames.append(y[["patient_id", "label_time", "label_code", "label_value"]])

    if not out_frames:
        return pd.DataFrame(columns=["patient_id", "label_time", "label_code", "label_value"])

    out = pd.concat(out_frames, ignore_index=True)
    out["patient_id"] = out["patient_id"].astype(str)
    out["label_time"] = pd.to_datetime(out["label_time"], errors="raise")
    out["label_code"] = out["label_code"].astype(str)
    return out[["patient_id", "label_time", "label_code", "label_value"]]


def main(raw_path: str, out_dir: str) -> None:
    raw_path_p = Path(raw_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    df_raw = _read_raw(raw_path_p)
    df_norm = _normalize_tjh(df_raw)

    dynamic = build_dynamic(df_norm)
    static = build_static(df_norm)
    label = build_label(df_norm)

    (out_dir_p / "dynamic.csv").write_text(dynamic.to_csv(index=False), encoding="utf-8")
    (out_dir_p / "static.csv").write_text(static.to_csv(index=False), encoding="utf-8")
    (out_dir_p / "label.csv").write_text(label.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert TJH raw Excel to OneEHR standard CSVs.")
    p.add_argument("--raw", required=True, help="Path to TJH raw Excel file")
    p.add_argument("--out-dir", required=True, help="Output directory for dynamic/static/label CSVs")
    args = p.parse_args()
    main(args.raw, args.out_dir)
