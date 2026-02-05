from __future__ import annotations

from pathlib import Path

import pandas as pd

from oneehr.config.schema import DatasetConfig
from oneehr.utils.imports import load_callable


def load_event_table(cfg: DatasetConfig) -> pd.DataFrame:
    if cfg.path is None:
        raise ValueError("dataset.path is required for file-based datasets.")
    path = Path(cfg.path)
    if cfg.file_type.lower() == "csv":
        df = pd.read_csv(path)
    elif cfg.file_type.lower() in {"xlsx", "excel"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dataset.file_type={cfg.file_type!r}")

    df = df.copy()
    if cfg.converter_fn is not None:
        # Converter is responsible for returning the unified OneEHR event schema.
        fn = load_callable(cfg.converter_fn)
        df = fn(df, cfg)
    else:
        # Label column is optional: users can provide labels via labels.fn.
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
