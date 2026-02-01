from __future__ import annotations

from dataclasses import dataclass

from oneehr.config.schema import ExperimentConfig
from oneehr.hpo.grid import apply_overrides, iter_grid


@dataclass(frozen=True)
class TrialResult:
    overrides: dict[str, object]
    score: float
    metrics: dict[str, float]


def select_best_overrides(
    cfg: ExperimentConfig,
    eval_trial,
) -> TrialResult | None:
    """Select best override dict using validation metrics.

    `eval_trial(cfg_trial)` must return (score, metrics).
    """

    best: TrialResult | None = None

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
        if best is None or better(score, best.score):
            best = TrialResult(overrides=dict(overrides), score=float(score), metrics=dict(metrics))

    return best

