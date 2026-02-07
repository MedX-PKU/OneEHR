from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    # Allow storing numpy/pandas objects in artifacts without callers having
    # to manually convert them to python scalars/lists.
    import numpy as np

    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    def _default(o: Any):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if pd is not None and isinstance(o, pd.Series):  # type: ignore[attr-defined]
            return o.to_dict()
        if pd is not None and isinstance(o, pd.DataFrame):  # type: ignore[attr-defined]
            return o.to_dict(orient="list")
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    path = Path(path)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_default) + "\n",
        encoding="utf-8",
    )
