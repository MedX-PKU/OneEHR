from __future__ import annotations

from typing import Any

import pandas as pd

from oneehr.data.splits import Split


def build_test_sample_frame(
    *,
    splits: list[Split],
    labels_df: pd.DataFrame | None,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    task_kind: str,
    prediction_mode: str,
) -> pd.DataFrame:
    if prediction_mode == "patient":
        return _build_patient_samples(
            splits=splits,
            labels_df=labels_df,
            dynamic=dynamic,
            static=static,
            task_kind=task_kind,
        )
    if prediction_mode == "time":
        return _build_time_samples(
            splits=splits,
            labels_df=labels_df,
            dynamic=dynamic,
            static=static,
            task_kind=task_kind,
        )
    raise ValueError(f"Unsupported prediction_mode={prediction_mode!r}")


def _build_patient_samples(
    *,
    splits: list[Split],
    labels_df: pd.DataFrame | None,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    task_kind: str,
) -> pd.DataFrame:
    label_map: dict[str, float] = {}
    if labels_df is not None and not labels_df.empty:
        labels = labels_df[["patient_id", "label"]].copy()
        labels["patient_id"] = labels["patient_id"].astype(str)
        labels = labels.drop_duplicates(subset=["patient_id"], keep="last")
        label_map = {
            str(pid): float(lbl)
            for pid, lbl in zip(labels["patient_id"], labels["label"])
        }

    dynamic_stats: dict[str, dict[str, object]] = {}
    if dynamic is not None and not dynamic.empty:
        tmp = dynamic.copy()
        tmp["patient_id"] = tmp["patient_id"].astype(str)
        tmp["event_time"] = pd.to_datetime(tmp["event_time"], errors="raise")
        grouped = tmp.groupby("patient_id", sort=False)
        dynamic_stats = {
            str(pid): {
                "event_count": int(len(group)),
                "first_event_time": group["event_time"].min(),
                "last_event_time": group["event_time"].max(),
            }
            for pid, group in grouped
        }

    static_ids = _static_patient_ids(static)
    rows: list[dict[str, Any]] = []
    for sp in splits:
        for patient_id in sp.test_patients.astype(str).tolist():
            stats = dynamic_stats.get(patient_id, {})
            rows.append(
                {
                    "sample_id": f"{sp.name}:{patient_id}",
                    "split": sp.name,
                    "split_role": "test",
                    "patient_id": patient_id,
                    "prediction_mode": "patient",
                    "task_kind": task_kind,
                    "ground_truth": label_map.get(patient_id),
                    "event_count": int(stats.get("event_count", 0)),
                    "first_event_time": stats.get("first_event_time"),
                    "last_event_time": stats.get("last_event_time"),
                    "has_static": patient_id in static_ids,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["split", "patient_id"],
        kind="stable",
    ).reset_index(drop=True)


def _build_time_samples(
    *,
    splits: list[Split],
    labels_df: pd.DataFrame | None,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    task_kind: str,
) -> pd.DataFrame:
    if dynamic is None or dynamic.empty:
        raise SystemExit("Time-window workflows require dataset.dynamic.")
    if labels_df is None or labels_df.empty:
        raise SystemExit("Time-window workflows require labels.parquet.")

    labels = labels_df.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels["bin_time"] = pd.to_datetime(labels["bin_time"], errors="raise")
    if "mask" in labels.columns:
        labels = labels[labels["mask"].astype(int) != 0].copy()

    static_ids = _static_patient_ids(static)
    rows: list[dict[str, Any]] = []
    for sp in splits:
        test_patients = set(sp.test_patients.astype(str).tolist())
        block = labels[labels["patient_id"].isin(test_patients)].copy()
        block = block.sort_values(["patient_id", "bin_time"], kind="stable")
        for _, row in block.iterrows():
            patient_id = str(row["patient_id"])
            bin_time = pd.to_datetime(row["bin_time"], errors="raise")
            rows.append(
                {
                    "sample_id": f"{sp.name}:{patient_id}:{bin_time.isoformat()}",
                    "split": sp.name,
                    "split_role": "test",
                    "patient_id": patient_id,
                    "bin_time": bin_time,
                    "prediction_mode": "time",
                    "task_kind": task_kind,
                    "ground_truth": float(row["label"]),
                    "has_static": patient_id in static_ids,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["split", "patient_id", "bin_time"],
        kind="stable",
    ).reset_index(drop=True)


def _static_patient_ids(static: pd.DataFrame | None) -> set[str]:
    if static is None or static.empty:
        return set()
    return set(static["patient_id"].astype(str).tolist())
