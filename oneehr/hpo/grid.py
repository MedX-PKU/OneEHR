from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from oneehr.config.schema import ExperimentConfig, HPOConfig


def _set_attr(obj: Any, path: list[str], value: Any) -> Any:
    if not path:
        return value
    head, *rest = path
    current = getattr(obj, head)
    updated = _set_attr(current, rest, value)
    return replace(obj, **{head: updated})


def apply_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        if parts[0] != "model" and parts[0] != "trainer" and parts[0] != "preprocess" and parts[0] != "split":
            raise ValueError(f"Unsupported override root: {parts[0]!r}")
        out = _set_attr(out, parts, value)
    return out


def iter_grid(hpo: HPOConfig):
    if not hpo.enabled or not hpo.grid:
        yield {}
        return

    keys = [k for k, _ in hpo.grid]
    values = [v for _, v in hpo.grid]

    def rec(i: int, cur: dict[str, Any]):
        if i == len(keys):
            yield dict(cur)
            return
        k = keys[i]
        for v in values[i]:
            cur[k] = v
            yield from rec(i + 1, cur)
        cur.pop(k, None)

    yield from rec(0, {})
