from __future__ import annotations

from dataclasses import dataclass

from oneehr.config.schema import ExperimentConfig
from oneehr.hpo.grid import apply_overrides, iter_grid


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
