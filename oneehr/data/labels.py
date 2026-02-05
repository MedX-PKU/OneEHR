from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from oneehr.config.schema import ExperimentConfig
from oneehr.utils.imports import load_callable
from oneehr.utils.time import parse_bin_size


class LabelFn(Protocol):
    def __call__(
        self,
        dynamic: pd.DataFrame,
        static: pd.DataFrame | None,
        label: pd.DataFrame | None,
        cfg: ExperimentConfig,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class LabelsResult:
    df: pd.DataFrame


def run_label_fn(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg: ExperimentConfig,
) -> LabelsResult | None:
    if cfg.labels.fn is None:
        return None
    fn = load_callable(cfg.labels.fn)
    out = fn(dynamic, static, label, cfg)
    if not isinstance(out, pd.DataFrame):
        raise TypeError("label_fn must return a pandas.DataFrame")
    return LabelsResult(df=out)


def normalize_patient_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "label"}
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"N-1 labels missing columns: {missing}")
    out = labels[["patient_id", "label"]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out = out.dropna(subset=["label"]).drop_duplicates(subset=["patient_id"], keep="last")
    return out


def normalize_time_labels(labels: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Normalize N-N label table to columns: patient_id, bin_time, label, mask."""

    if "patient_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError("N-N labels must contain patient_id and label")

    out = labels.copy()
    out["patient_id"] = out["patient_id"].astype(str)

    if "bin_time" not in out.columns:
        if not cfg.labels.bin_from_time_col:
            raise ValueError("N-N labels missing bin_time and bin_from_time_col is false")
        if "label_time" not in out.columns:
            raise ValueError(
                "N-N labels missing bin_time and required time column 'label_time' for binning"
            )
        out["label_time"] = pd.to_datetime(out["label_time"], errors="raise")
        freq = parse_bin_size(cfg.preprocess.bin_size)
        out["bin_time"] = out["label_time"].dt.floor(freq)

    out["bin_time"] = pd.to_datetime(out["bin_time"], errors="raise")

    if "mask" not in out.columns:
        out["mask"] = 1

    out = out[["patient_id", "bin_time", "label", "mask"]].copy()
    out = out.dropna(subset=["label"])
    out["mask"] = out["mask"].astype(int)
    out = out.drop_duplicates(subset=["patient_id", "bin_time"], keep="last")
    return out
