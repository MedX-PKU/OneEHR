from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


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
