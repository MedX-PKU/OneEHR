from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    dynamic: Path | None = None
    static: Path | None = None
    label: Path | None = None


@dataclass(frozen=True)
class PreprocessConfig:
    bin_size: str = "1d"
    numeric_strategy: str = "mean"  # mean | last | median | min | max | std | count
    categorical_strategy: str = "onehot"  # onehot | count
    code_selection: str = "frequency"  # frequency | all | list
    top_k_codes: int = 100
    min_code_count: int = 1
    max_seq_length: int | None = None  # truncate to most recent N bins
    min_events_per_patient: int = 1  # exclude patients with fewer events
    pipeline: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class TaskConfig:
    kind: str = "binary"  # binary | regression | multiclass
    prediction_mode: str = "patient"  # patient | time
    num_classes: int | None = None  # required when kind="multiclass"
    loss: str = "default"  # default | focal
    focal_gamma: float = 2.0  # gamma for focal loss


@dataclass(frozen=True)
class SplitConfig:
    kind: str = "random"  # random | time
    seed: int = 42
    val_size: float = 0.1
    test_size: float = 0.2
    time_boundary: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str = "xgboost"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerConfig:
    device: str = "auto"
    seed: int = 42
    max_epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 0
    precision: str = "fp32"  # fp32 | fp16 | bf16
    early_stopping: bool = True
    patience: int = 5
    monitor: str = "val_loss"  # val_loss | val_auroc | val_auprc | val_rmse | val_mae
    monitor_mode: str = "min"  # min | max (auto-set for known metrics)
    scheduler: str = "none"  # none | cosine | step | plateau
    scheduler_params: dict[str, object] = field(default_factory=dict)
    class_weight: str = "none"  # none | balanced


@dataclass(frozen=True)
class SystemConfig:
    name: str = ""
    kind: str = "llm"  # llm | agent
    framework: str = "single_llm"
    backend: str = "openai"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("runs")
    run_name: str = "exp001"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: list[ModelConfig] = field(default_factory=list)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    systems: list[SystemConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)

    def run_dir(self) -> Path:
        return self.output.root / self.output.run_name
