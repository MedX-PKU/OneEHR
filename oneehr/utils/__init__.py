"""Shared utility functions (merged from io.py, imports.py, time.py)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from types import ModuleType
from typing import Any


# ─── Reproducibility ─────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── IO helpers ───────────────────────────────────────────────────────────────


def as_jsonable(v: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    try:
        import numpy as np
        import pandas as pd

        if isinstance(v, np.ndarray):
            return as_jsonable(v.tolist())
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return as_jsonable(v.item())
        if isinstance(v, pd.Series):
            return {str(k): as_jsonable(x) for k, x in v.to_dict().items()}
        if isinstance(v, pd.DataFrame):
            return {str(k): as_jsonable(x) for k, x in v.to_dict(orient="list").items()}
    except Exception:
        pass
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


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(as_jsonable(row), sort_keys=True))
            fh.write("\n")


# ─── Import helpers ───────────────────────────────────────────────────────────


def load_module_from_path(path: str | Path, module_name: str) -> ModuleType:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_callable(ref: str):
    """Load a callable from `path/to/file.py:func_name`."""
    if ":" not in ref:
        raise ValueError(f"Invalid callable ref: {ref!r}. Expected 'file.py:func'.")
    file_path, func_name = ref.split(":", 1)
    module = load_module_from_path(file_path, module_name=f"oneehr_user_{Path(file_path).stem}")
    fn = getattr(module, func_name, None)
    if fn is None:
        raise AttributeError(f"Function {func_name!r} not found in {file_path!r}")
    if not callable(fn):
        raise TypeError(f"{ref!r} is not callable")
    return fn


# ─── Time helpers ─────────────────────────────────────────────────────────────


_BIN_RE = re.compile(r"^(\d+)([smhdw])$")


def parse_bin_size(bin_size: str) -> str:
    """Convert human-friendly bin size like `6h` to a pandas-compatible offset alias."""
    m = _BIN_RE.match(bin_size.strip().lower())
    if not m:
        raise ValueError(f"Invalid bin_size: {bin_size!r}. Expected like '1h', '6h', '1d'.")
    n = int(m.group(1))
    unit = m.group(2)
    unit_map = {"s": "S", "m": "min", "h": "H", "d": "D", "w": "W"}
    return f"{n}{unit_map[unit]}"
