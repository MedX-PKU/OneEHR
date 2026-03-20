"""Simplified grid search HPO (merged from hpo/grid.py + hpo/runner.py)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
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
        if parts[0] not in {"model", "trainer", "preprocess", "split"}:
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


@dataclass(frozen=True)
class TrialResult:
    overrides: dict[str, object]
    score: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class BestResult:
    best: TrialResult | None
    trials: list[TrialResult]


def select_best_with_trials(cfg: ExperimentConfig, eval_trial) -> BestResult:
    best: TrialResult | None = None
    trials: list[TrialResult] = []

    def better(a: float, b: float) -> bool:
        if cfg.hpo.mode == "min":
            return a < b
        if cfg.hpo.mode == "max":
            return a > b
        raise ValueError(f"Unsupported hpo.mode={cfg.hpo.mode!r}")

    for overrides in iter_grid(cfg.hpo):
        cfg_trial = apply_overrides(cfg, overrides)
        res = eval_trial(cfg_trial)
        if res is None:
            continue
        score, metrics = res
        tr = TrialResult(overrides=dict(overrides), score=float(score), metrics=dict(metrics))
        trials.append(tr)
        if best is None or better(tr.score, best.score):
            best = tr

    return BestResult(best=best, trials=trials)
