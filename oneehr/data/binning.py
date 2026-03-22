from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.config.schema import PreprocessConfig
from oneehr.utils import parse_bin_size


@dataclass(frozen=True)
class BinnedTable:
    table: pd.DataFrame
    code_vocab: list[str]
    feature_schema: list[dict]  # [{"name": code, "cols": [...], "dim": int}, ...]
    obs_mask: pd.DataFrame  # boolean mask (same shape as feature cols), pre-fillna


def _infer_value_type(series: pd.Series) -> str:
    # Heuristic: if >90% values are numeric-coercible => numeric.
    coerced = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(coerced.notna().mean())
    return "numeric" if numeric_ratio >= 0.9 else "categorical"


def _load_importance_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported importance file type: {path.suffix}")


def _select_code_vocab(
    df: pd.DataFrame,
    preprocess: PreprocessConfig,
) -> list[str]:
    code_counts = df["code"].value_counts()
    code_counts = code_counts[code_counts >= preprocess.min_code_count]

    selection = preprocess.code_selection
    if selection == "all":
        return list(code_counts.index.astype(str))
    if selection == "frequency":
        top_k = preprocess.top_k_codes
        if top_k is None:
            return list(code_counts.index.astype(str))
        return list(code_counts.head(top_k).index.astype(str))
    if selection == "list":
        code_list = getattr(preprocess, "code_list", [])
        if not code_list:
            raise ValueError("preprocess.code_list must be provided when code_selection='list'.")
        return [str(c) for c in code_list]
    raise ValueError(f"Unsupported preprocess.code_selection={selection!r}")


def _normalize_cat_value(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "__nan__"
    s = str(v).strip()
    return s if s else "__nan__"


def bin_events(
    events: pd.DataFrame,
    dataset: object,
    preprocess: PreprocessConfig,
) -> BinnedTable:
    """Convert event-level table into bin-level table.

    Output is a patient-by-bin table in *long* format with one row per (patient_id, bin_time).
    Features are expanded into columns using a limited code vocabulary.
    """

    cols = [
        "patient_id",
        "event_time",
        "code",
        "value",
    ]
    missing = [c for c in cols if c not in events.columns]
    if missing:
        raise ValueError(f"dynamic.csv missing required columns for binning: {missing}")
    df = events[cols].copy()

    # Determine bin start time.
    freq = parse_bin_size(preprocess.bin_size)
    df["bin_time"] = df["event_time"].dt.floor(freq)

    # Build code vocab based on selection strategy.
    code_vocab = _select_code_vocab(df, preprocess)
    if preprocess.code_selection != "list":
        code_counts = df["code"].value_counts()
        code_vocab = [c for c in code_vocab if c in code_counts.index.astype(str)]
    df = df[df["code"].isin(code_vocab)].copy()

    if df.empty:
        raise ValueError(
            "No events left after applying code vocab filters. "
            "Try adjusting `code_selection`, increasing `top_k_codes`, or decreasing `min_code_count`."
        )

    # Filter patients with too few events.
    min_events = preprocess.min_events_per_patient
    if min_events > 1:
        event_counts = df.groupby("patient_id").size()
        keep_pids = event_counts[event_counts >= min_events].index
        n_before = df["patient_id"].nunique()
        df = df[df["patient_id"].isin(keep_pids)].copy()
        n_after = df["patient_id"].nunique()
        if n_after < n_before:
            import warnings
            warnings.warn(
                f"min_events_per_patient={min_events}: dropped {n_before - n_after} patients "
                f"({n_before} → {n_after})",
                stacklevel=2,
            )
        if df.empty:
            raise ValueError(
                f"No patients left after min_events_per_patient={min_events} filter."
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
        strategy = preprocess.numeric_strategy
        grp = numeric.groupby(["patient_id", "bin_time", "code"], sort=False)["value_num"]
        if strategy == "mean":
            agg = grp.mean().reset_index()
        elif strategy == "last":
            numeric = numeric.sort_values(["patient_id", "event_time"], kind="stable")
            grp = numeric.groupby(["patient_id", "bin_time", "code"], sort=False)["value_num"]
            agg = grp.last().reset_index()
        elif strategy == "median":
            agg = grp.median().reset_index()
        elif strategy == "min":
            agg = grp.min().reset_index()
        elif strategy == "max":
            agg = grp.max().reset_index()
        elif strategy == "std":
            agg = grp.std().reset_index()
            agg["value_num"] = agg["value_num"].fillna(0.0)
        elif strategy == "count":
            agg = grp.count().reset_index()
        else:
            raise ValueError(
                f"Unsupported preprocess.numeric_strategy={strategy!r}. "
                "Expected one of: mean, last, median, min, max, std, count."
            )

        wide = agg.pivot(index=["patient_id", "bin_time"], columns="code", values="value_num")
        wide = wide.add_prefix("num__").reset_index()
        feats.append(wide)

    if categorical_codes:
        cat = df[df["code"].isin(categorical_codes)].copy()
        if preprocess.categorical_strategy == "onehot":
            # Represent categorical events as one-hot over (code, value) per bin:
            #   cat__{code}__{value} in {0,1}
            # If a (code,value) appears multiple times in a bin, it is still 1.0.
            cat["_present"] = 1.0
            cat["cat_key"] = cat["code"].astype(str) + "__" + cat["value"].map(_normalize_cat_value)
            wide = cat.pivot_table(
                index=["patient_id", "bin_time"],
                columns="cat_key",
                values="_present",
                aggfunc="max",
                fill_value=0.0,
            )
            wide.columns = [f"cat__{c}" for c in wide.columns.astype(str)]
            wide = wide.reset_index()
            feats.append(wide)
        elif preprocess.categorical_strategy == "count":
            # Per-bin count of events for that code.
            agg = (
                cat.groupby(["patient_id", "bin_time", "code"], sort=False)
                .size()
                .rename("count")
                .reset_index()
            )
            wide = agg.pivot(index=["patient_id", "bin_time"], columns="code", values="count")
            wide = wide.fillna(0.0).add_prefix("cat__").reset_index()
            feats.append(wide)
        else:
            raise ValueError(
                f"Unsupported preprocess.categorical_strategy={preprocess.categorical_strategy!r}. "
                "Expected 'onehot' or 'count'."
            )

    # Merge numeric + categorical features.
    out = feats[0]
    for f in feats[1:]:
        out = out.merge(f, on=["patient_id", "bin_time"], how="outer")

    out = out.sort_values(["patient_id", "bin_time"], kind="stable")

    # Attach label if provided in the raw events table.
    if "label" in df.columns:
        labels = (
            df[["patient_id", "label"]]
            .dropna(subset=["label"])
            .drop_duplicates(subset=["patient_id"], keep="last")
        )
        out = out.merge(labels, on="patient_id", how="left")
    else:
        out["label"] = np.nan

    # Build feature_schema: map each original code to its encoded columns.
    feat_cols = [c for c in out.columns if c.startswith("num__") or c.startswith("cat__")]
    feature_schema: list[dict] = []
    for code in code_vocab:
        cols = [c for c in feat_cols if c == f"num__{code}" or c.startswith(f"cat__{code}__")]
        if cols:
            feature_schema.append({"name": code, "cols": cols, "dim": len(cols)})

    # Compute observation mask BEFORE filling NaN — captures true missingness.
    obs_mask = out[feat_cols].notna().astype(np.float32)
    # Keep patient_id + bin_time alongside for alignment.
    id_cols = [c for c in ["patient_id", "bin_time"] if c in out.columns]
    obs_mask = pd.concat([out[id_cols].reset_index(drop=True), obs_mask.reset_index(drop=True)], axis=1)

    # Fill missing categorical features with 0 (absence indicator).
    # Numeric features keep NaN — imputation is handled by the pipeline.
    for col in out.columns:
        if col.startswith("cat__"):
            out[col] = out[col].fillna(0.0)

    return BinnedTable(table=out, code_vocab=code_vocab, feature_schema=feature_schema, obs_mask=obs_mask)
