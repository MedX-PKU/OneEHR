from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ConvertedDataset:
    """Standard return type for dataset converters.

    Converters are intended to be small, dataset-specific scripts (often living
    outside the OneEHR repo) that adapt arbitrary raw formats into the unified
    event table.

    Required:
    - `events`: unified event table (long format)

    Optional:
    - `meta`: any extra debugging info (ignored by the pipeline)
    """

    events: pd.DataFrame
    meta: dict[str, Any] | None = None

