from __future__ import annotations

from pathlib import Path

import pandas as pd

from oneehr.config.schema import DatasetConfig
from oneehr.utils.imports import load_callable


def load_event_table(cfg: DatasetConfig) -> pd.DataFrame:
    if cfg.path is None:
        raise ValueError("dataset.path is required.")
    path = Path(cfg.path)
    if cfg.converter_fn is not None:
        # For converter-based datasets, let the converter own raw IO details.
        # We still provide a small convenience: read Excel/CSV based on suffix.
        suf = path.suffix.lower()
        if suf in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suf == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported raw dataset file type for converter: {path.suffix}. "
                "Use .csv/.xlsx or extend the converter to handle it."
            )
    else:
        if cfg.file_type.lower() == "csv":
            df = pd.read_csv(path)
        elif cfg.file_type.lower() in {"xlsx", "excel"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported dataset.file_type={cfg.file_type!r}")

    df = df.copy()
    if cfg.converter_fn is not None:
        fn = load_callable(cfg.converter_fn)
        res = fn(df, cfg)
        # converter may return DataFrame (events) or an object/dict with events/labels.
        if isinstance(res, pd.DataFrame):
            events = res
        elif isinstance(res, dict) and "events" in res:
            events = res["events"]
        elif hasattr(res, "events"):
            events = res.events
        else:
            raise TypeError("converter_fn must return a DataFrame, or an object with `.events`")

        # Basic validation + time parsing.
        required = [cfg.patient_id_col, cfg.time_col, cfg.code_col, cfg.value_col]
        missing = [c for c in required if c not in events.columns]
        if missing:
            raise ValueError(f"converter output missing required columns: {missing}")
        events = events.copy()
        events[cfg.time_col] = pd.to_datetime(events[cfg.time_col], errors="raise")
        return events

    # No converter: expect already-normalized event table.
    required = [cfg.patient_id_col, cfg.code_col, cfg.value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got columns={list(df.columns)}")

    if cfg.time_col not in df.columns:
        raise ValueError(
            f"Missing time column {cfg.time_col!r}. MVP requires a parseable time column for fixed binning."
        )

    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], format=cfg.time_format, errors="raise")
    return df
