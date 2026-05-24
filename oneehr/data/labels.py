from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from oneehr.config.schema import ExperimentConfig
from oneehr.utils import load_callable, parse_bin_size


class LabelFn:
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
    *,
    label_fn_ref: str | None = None,
) -> LabelsResult | None:
    """Run a user-supplied label function, if provided."""
    if label_fn_ref is None:
        return None
    fn = load_callable(label_fn_ref)
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


def normalize_multilabel_patient_labels(labels: pd.DataFrame, *, num_classes: int | None = None) -> pd.DataFrame:
    required = {"patient_id", "label_code", "label"}
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"Multi-label N-1 labels missing columns: {missing}")

    out = labels.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["label_code"] = out["label_code"].astype(str)
    sort_cols = [c for c in ("patient_id", "label_time", "label_code") if c in out.columns]
    out = out.sort_values(sort_cols, kind="stable").dropna(subset=["label"])
    out = out.drop_duplicates(subset=["patient_id", "label_code"], keep="last")
    return _pivot_multilabel(out, index_cols=["patient_id"], num_classes=num_classes)


def normalize_time_labels(labels: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if "patient_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError("N-N labels must contain patient_id and label")

    out = labels.copy()
    out["patient_id"] = out["patient_id"].astype(str)

    if "bin_time" not in out.columns:
        if "label_time" not in out.columns:
            raise ValueError("N-N labels missing bin_time and label_time")
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


def normalize_multilabel_time_labels(labels: pd.DataFrame, cfg: ExperimentConfig, *, num_classes: int | None = None) -> pd.DataFrame:
    required = {"patient_id", "label_code", "label"}
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"Multi-label N-N labels missing columns: {missing}")

    out = labels.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["label_code"] = out["label_code"].astype(str)

    if "bin_time" not in out.columns:
        if "label_time" not in out.columns:
            raise ValueError("Multi-label N-N labels missing bin_time and label_time")
        out["label_time"] = pd.to_datetime(out["label_time"], errors="raise")
        freq = parse_bin_size(cfg.preprocess.bin_size)
        out["bin_time"] = out["label_time"].dt.floor(freq)

    out["bin_time"] = pd.to_datetime(out["bin_time"], errors="raise")
    if "mask" not in out.columns:
        out["mask"] = 1

    sort_cols = [c for c in ("patient_id", "bin_time", "label_time", "label_code") if c in out.columns]
    out = out.sort_values(sort_cols, kind="stable").dropna(subset=["label"])
    out = out.drop_duplicates(subset=["patient_id", "bin_time", "label_code"], keep="last")
    mask = out.groupby(["patient_id", "bin_time"], sort=False)["mask"].max().astype(int).reset_index()
    pivot = _pivot_multilabel(out, index_cols=["patient_id", "bin_time"], num_classes=num_classes)
    return pivot.merge(mask, on=["patient_id", "bin_time"], how="left").fillna({"mask": 1})


def _pivot_multilabel(labels: pd.DataFrame, *, index_cols: list[str], num_classes: int | None) -> pd.DataFrame:
    code_order = sorted(labels["label_code"].astype(str).unique().tolist())
    if num_classes is not None and len(code_order) != int(num_classes):
        raise ValueError(f"task.num_classes={num_classes} but found {len(code_order)} unique label_code values")

    pivot = labels.pivot_table(
        index=index_cols,
        columns="label_code",
        values="label",
        aggfunc="last",
        fill_value=0.0,
        sort=False,
    )
    pivot = pivot.reindex(columns=code_order, fill_value=0.0)
    safe_cols = _safe_label_columns(code_order)
    pivot.columns = safe_cols
    return pivot.reset_index()


def _safe_label_columns(codes: list[str]) -> list[str]:
    cols = []
    seen: dict[str, int] = {}
    for i, code in enumerate(codes):
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", code).strip("_") or str(i)
        col = f"label_{i}_{safe}"
        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
        cols.append(col)
    return cols
