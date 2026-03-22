from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from oneehr.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    PreprocessConfig,
    SplitConfig,
    SystemConfig,
    TaskConfig,
    TrainerConfig,
)


def _build_dataclass(cls: type, raw: dict[str, Any]) -> Any:
    """Generic constructor: pick only fields the dataclass accepts."""
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    unknown = [k for k in raw if k not in fields]
    if unknown:
        warnings.warn(
            f"Unknown config keys for [{cls.__name__}]: {unknown} — ignored",
            stacklevel=3,
        )
    for key, val in raw.items():
        if key not in fields:
            continue
        f = next(f for f in dataclasses.fields(cls) if f.name == key)
        # Convert Path fields
        if f.type in ("Path | None", "Path"):
            kwargs[key] = Path(val) if val not in (None, "") else None
        else:
            kwargs[key] = val
    return cls(**kwargs)


def _validate_config(cfg: ExperimentConfig) -> None:
    """Validate config values and raise on errors, warn on suspicious values."""
    errors: list[str] = []
    warns: list[str] = []

    # --- Dataset paths (warn only — paths are checked at preprocess time) ---
    if cfg.dataset.dynamic is not None and not cfg.dataset.dynamic.exists():
        warns.append(f"dataset.dynamic path does not exist: {cfg.dataset.dynamic}")
    if cfg.dataset.static is not None and not cfg.dataset.static.exists():
        warns.append(f"dataset.static path does not exist: {cfg.dataset.static}")
    if cfg.dataset.label is not None and not cfg.dataset.label.exists():
        warns.append(f"dataset.label path does not exist: {cfg.dataset.label}")

    # --- Task ---
    valid_kinds = ("binary", "regression", "multiclass", "survival", "multilabel")
    if cfg.task.kind not in valid_kinds:
        errors.append(f"task.kind={cfg.task.kind!r} — expected one of {valid_kinds}")
    if cfg.task.kind == "multiclass" and (cfg.task.num_classes is None or cfg.task.num_classes < 2):
        errors.append("task.num_classes must be >= 2 when kind='multiclass'")
    valid_modes = ("patient", "time")
    if cfg.task.prediction_mode not in valid_modes:
        errors.append(f"task.prediction_mode={cfg.task.prediction_mode!r} — expected one of {valid_modes}")

    # --- Split ---
    s = cfg.split
    if not (0 < s.val_size < 1):
        errors.append(f"split.val_size={s.val_size} — must be in (0, 1)")
    if not (0 < s.test_size < 1):
        errors.append(f"split.test_size={s.test_size} — must be in (0, 1)")
    if s.val_size + s.test_size >= 1:
        errors.append(f"split.val_size + test_size = {s.val_size + s.test_size} — must be < 1")

    # --- Trainer ---
    t = cfg.trainer
    if t.lr <= 0 or t.lr >= 1:
        errors.append(f"trainer.lr={t.lr} — must be in (0, 1)")
    if t.batch_size <= 0:
        errors.append(f"trainer.batch_size={t.batch_size} — must be > 0")
    if t.max_epochs <= 0:
        errors.append(f"trainer.max_epochs={t.max_epochs} — must be > 0")
    if t.patience <= 0:
        errors.append(f"trainer.patience={t.patience} — must be > 0")
    if t.early_stopping and s.val_size <= 0:
        errors.append("trainer.early_stopping=true requires split.val_size > 0")
    if t.lr > 0.1:
        warns.append(f"trainer.lr={t.lr} is unusually high — typical range is [1e-5, 1e-2]")
    valid_precision = ("fp32", "fp16", "bf16")
    if t.precision not in valid_precision:
        errors.append(f"trainer.precision={t.precision!r} — expected one of {valid_precision}")
    valid_schedulers = ("none", "cosine", "step", "plateau")
    if t.scheduler not in valid_schedulers:
        errors.append(f"trainer.scheduler={t.scheduler!r} — expected one of {valid_schedulers}")

    # --- Preprocess ---
    p = cfg.preprocess
    valid_num = ("mean", "last", "median", "min", "max", "std", "count")
    if p.numeric_strategy not in valid_num:
        errors.append(f"preprocess.numeric_strategy={p.numeric_strategy!r} — expected one of {valid_num}")
    valid_cat = ("onehot", "count")
    if p.categorical_strategy not in valid_cat:
        errors.append(f"preprocess.categorical_strategy={p.categorical_strategy!r} — expected one of {valid_cat}")
    if p.top_k_codes is not None and p.top_k_codes <= 0:
        errors.append(f"preprocess.top_k_codes={p.top_k_codes} — must be > 0")
    if p.max_seq_length is not None and p.max_seq_length <= 0:
        errors.append(f"preprocess.max_seq_length={p.max_seq_length} — must be > 0")
    if p.min_events_per_patient < 1:
        errors.append(f"preprocess.min_events_per_patient={p.min_events_per_patient} — must be >= 1")

    for w in warns:
        warnings.warn(f"Config warning: {w}", stacklevel=3)
    if errors:
        msg = "Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    dataset = _build_dataclass(DatasetConfig, raw.get("dataset", {}))
    preprocess = _build_dataclass(PreprocessConfig, raw.get("preprocess", {}))
    task = _build_dataclass(TaskConfig, raw.get("task", {}))
    split = _build_dataclass(SplitConfig, raw.get("split", {}))
    trainer = _build_dataclass(TrainerConfig, raw.get("trainer", {}))
    output = _build_dataclass(OutputConfig, raw.get("output", {}))

    models: list[ModelConfig] = []
    for m in raw.get("models", []):
        name = m.get("name", "xgboost")
        params = m.get("params", {})
        models.append(ModelConfig(name=name, params=dict(params)))

    systems: list[SystemConfig] = []
    for s in raw.get("systems", []):
        params = s.pop("params", {}) if "params" in s else {}
        sc = _build_dataclass(SystemConfig, s)
        if params:
            sc = dataclasses.replace(sc, params=dict(params))
        systems.append(sc)

    cfg = ExperimentConfig(
        dataset=dataset,
        preprocess=preprocess,
        task=task,
        split=split,
        models=models,
        trainer=trainer,
        systems=systems,
        output=output,
    )

    _validate_config(cfg)

    return cfg
