from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from oneehr.config.schema import DatasetConfig
from oneehr.datasets.tjh import load_tjh_events


_LOADERS: dict[str, Callable[[DatasetConfig], pd.DataFrame]] = {
    "tjh": load_tjh_events,
}


def list_datasets() -> list[str]:
    return sorted(_LOADERS.keys())


def load_events(cfg: DatasetConfig) -> pd.DataFrame:
    """Load a dataset as a normalized event table.

    If `cfg.name` is set, dispatch to a built-in adapter.
    Otherwise, fallback to the default CSV/XLSX loader (see oneehr.data.io).
    """

    if cfg.name is None:
        from oneehr.data.io import load_event_table

        return load_event_table(cfg)

    name = str(cfg.name).strip().lower()
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset.name={cfg.name!r}. Available={list_datasets()}"
        )
    return _LOADERS[name](cfg)

