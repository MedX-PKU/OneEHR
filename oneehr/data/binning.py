from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneehr.config.schema import DatasetConfig, PreprocessConfig
from oneehr.utils.time import parse_bin_size


@dataclass(frozen=True)
class BinnedTable:
    table: pd.DataFrame
    code_vocab: list[str]


def _infer_value_type(series: pd.Series) -> str:
    # Heuristic: if >90% values are numeric-coercible => numeric.
    coerced = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(coerced.notna().mean())
    return "numeric" if numeric_ratio >= 0.9 else "categorical"


def bin_events(
    events: pd.DataFrame,
    dataset: DatasetConfig,
    preprocess: PreprocessConfig,
) -> BinnedTable:
    """Convert event-level table into bin-level table.

    Output is a patient-by-bin table in *long* format with one row per (patient_id, bin_time).
    Features are expanded into columns using a limited code vocabulary.
    """

    df = events[[
        dataset.patient_id_col,
        dataset.time_col,
        dataset.code_col,
        dataset.value_col,
        dataset.label_col,
    ]].copy()
    df.rename(
        columns={
            dataset.patient_id_col: "patient_id",
            dataset.time_col: "event_time",
            dataset.code_col: "code",
            dataset.value_col: "value",
            dataset.label_col: "label",
        },
        inplace=True,
    )

    # Determine bin start time.
    freq = parse_bin_size(preprocess.bin_size)
    df["bin_time"] = df["event_time"].dt.floor(freq)

    # Build code vocab using overall frequency.
    code_counts = df["code"].value_counts()
    code_counts = code_counts[code_counts >= preprocess.min_code_count]
    code_vocab = list(code_counts.head(preprocess.top_k_codes).index.astype(str))
    df = df[df["code"].isin(code_vocab)].copy()

    if df.empty:
        raise ValueError(
            "No events left after applying code vocab filters. "
            "Try increasing `top_k_codes` or decreasing `min_code_count`."
        )

    # Infer type per code.
    code_types: dict[str, str] = {}
    for code, g in df.groupby("code", sort=False):
        code_types[str(code)] = _infer_value_type(g["value"])

    numeric_codes = [c for c in code_vocab if code_types.get(c) == "numeric"]
    categorical_codes = [c for c in code_vocab if code_types.get(c) == "categorical"]

    feats: list[pd.DataFrame] = []

    if numeric_codes:
        numeric = df[df["code"].isin(numeric_codes)].copy()
        numeric["value_num"] = pd.to_numeric(numeric["value"], errors="coerce")
        numeric = numeric.dropna(subset=["value_num"])
        if preprocess.numeric_strategy == "mean":
            agg = (
                numeric.groupby(["patient_id", "bin_time", "code"], sort=False)["value_num"]
                .mean()
                .reset_index()
            )
        elif preprocess.numeric_strategy == "last":
            numeric = numeric.sort_values(["patient_id", "event_time"], kind="stable")
            agg = (
                numeric.groupby(["patient_id", "bin_time", "code"], sort=False)["value_num"]
                .last()
                .reset_index()
            )
        else:
            raise ValueError(f"Unsupported preprocess.numeric_strategy={preprocess.numeric_strategy!r}")

        wide = agg.pivot(index=["patient_id", "bin_time"], columns="code", values="value_num")
        wide = wide.add_prefix("num__").reset_index()
        feats.append(wide)

    if categorical_codes:
        cat = df[df["code"].isin(categorical_codes)].copy()
        # For MVP: per-bin count of events for that code.
        agg = (
            cat.groupby(["patient_id", "bin_time", "code"], sort=False)
            .size()
            .rename("count")
            .reset_index()
        )
        wide = agg.pivot(index=["patient_id", "bin_time"], columns="code", values="count")
        wide = wide.fillna(0.0).add_prefix("cat__").reset_index()
        feats.append(wide)

    # Merge numeric + categorical features.
    out = feats[0]
    for f in feats[1:]:
        out = out.merge(f, on=["patient_id", "bin_time"], how="outer")

    out = out.sort_values(["patient_id", "bin_time"], kind="stable")

    # Attach label.
    labels = (
        df[["patient_id", "label"]]
        .dropna(subset=["label"])
        .drop_duplicates(subset=["patient_id"], keep="last")
    )
    out = out.merge(labels, on="patient_id", how="left")

    # Fill missing numeric features with nan, categorical with 0.
    for col in out.columns:
        if col.startswith("cat__"):
            out[col] = out[col].fillna(0.0)
        elif col.startswith("num__"):
            # For DL sequence models, NaNs will poison training. Default to 0.
            out[col] = out[col].fillna(0.0)

    return BinnedTable(table=out, code_vocab=code_vocab)
