from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def as_jsonable(v: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [as_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): as_jsonable(x) for k, x in v.items()}
    return str(v)


def sha256_lines(lines: list[str]) -> str:
    """SHA-256 hash of a list of strings, with normalized whitespace."""
    norm = "\n".join([ln.strip() for ln in lines]) + "\n"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    # Allow storing numpy/pandas objects in artifacts without callers having
    # to manually convert them to python scalars/lists.
    import numpy as np
    import pandas as pd

    def _default(o: Any):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="list")
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    path = Path(path)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_default) + "\n",
        encoding="utf-8",
    )
