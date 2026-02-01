from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from oneehr.config.schema import ExperimentConfig
from oneehr.utils.imports import load_callable


class LabelFn(Protocol):
    def __call__(self, events: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame: ...


@dataclass(frozen=True)
class LabelsResult:
    df: pd.DataFrame


def run_label_fn(events: pd.DataFrame, cfg: ExperimentConfig) -> LabelsResult | None:
    if cfg.labels.fn is None:
        return None
    fn = load_callable(cfg.labels.fn)
    out = fn(events, cfg)
    if not isinstance(out, pd.DataFrame):
        raise TypeError("label_fn must return a pandas.DataFrame")
    return LabelsResult(df=out)
