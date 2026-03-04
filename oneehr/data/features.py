"""Feature column extraction utilities."""

from __future__ import annotations

import pandas as pd


def dynamic_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns whose names start with ``num__`` or ``cat__``."""
    return [c for c in df.columns if c.startswith("num__") or c.startswith("cat__")]


def has_static_branch(model: object) -> bool:
    """Check whether a model instance supports a dedicated static branch."""
    return hasattr(model, "static_dim") and int(getattr(model, "static_dim", 0)) > 0
