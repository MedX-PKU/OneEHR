from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatientLabels:
    patient_id: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class TimeLabels:
    patient_id: np.ndarray
    bin_time: np.ndarray
    y: np.ndarray
    mask: np.ndarray


def extract_patient_labels(binned: pd.DataFrame) -> PatientLabels:
    if "patient_id" not in binned.columns or "label" not in binned.columns:
        raise ValueError("binned must contain patient_id and label")
    lab = (
        binned[["patient_id", "label"]]
        .dropna(subset=["label"])
        .drop_duplicates(subset=["patient_id"], keep="last")
        .sort_values("patient_id", kind="stable")
    )
    return PatientLabels(patient_id=lab["patient_id"].astype(str).to_numpy(), y=lab["label"].to_numpy())


def make_time_labels_from_patient_label(binned: pd.DataFrame) -> TimeLabels:
    """Baseline N-N: broadcast patient-level label to each bin.

    This is *not* always clinically meaningful but serves as a working default.
    Users can override by providing a label_fn later.
    """

    if "patient_id" not in binned.columns or "bin_time" not in binned.columns or "label" not in binned.columns:
        raise ValueError("binned must contain patient_id, bin_time, label")
    df = binned[["patient_id", "bin_time", "label"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.dropna(subset=["label"])
    df = df.sort_values(["patient_id", "bin_time"], kind="stable")
    y = df["label"].to_numpy()
    mask = np.ones_like(y, dtype=bool)
    return TimeLabels(
        patient_id=df["patient_id"].to_numpy(),
        bin_time=df["bin_time"].to_numpy(),
        y=y,
        mask=mask,
    )

