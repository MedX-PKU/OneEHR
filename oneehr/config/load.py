from __future__ import annotations

import dataclasses
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


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
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

    return ExperimentConfig(
        dataset=dataset,
        preprocess=preprocess,
        task=task,
        split=split,
        models=models,
        trainer=trainer,
        systems=systems,
        output=output,
    )
