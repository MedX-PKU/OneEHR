from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    file_type: str = "csv"  # csv | xlsx
    patient_id_col: str = "patient_id"
    time_col: str = "event_time"
    code_col: str = "code"
    value_col: str = "value"
    label_col: str = "label"
    time_format: str | None = None


@dataclass(frozen=True)
class PreprocessConfig:
    bin_size: str = "1d"  # e.g. 1h, 6h, 1d
    numeric_strategy: str = "mean"  # mean | last
    categorical_strategy: str = "count"  # count
    code_selection: str = "frequency"  # frequency | all | list | importance
    top_k_codes: int | None = 500
    min_code_count: int = 1
    code_list: list[str] = field(default_factory=list)
    importance_file: Path | None = None
    importance_code_col: str = "code"
    importance_value_col: str = "importance"


@dataclass(frozen=True)
class TaskConfig:
    kind: str  # binary | regression
    prediction_mode: str = "patient"  # patient | time


@dataclass(frozen=True)
class LabelsConfig:
    fn: str | None = None  # e.g. path/to/label_fn.py:build_labels
    time_col: str = "label_time"
    bin_from_time_col: bool = True


@dataclass(frozen=True)
class SplitConfig:
    kind: str  # kfold | random | time
    seed: int = 42
    n_splits: int = 5
    val_size: float = 0.1
    test_size: float = 0.2
    time_boundary: str | None = None  # datetime string used for time split


@dataclass(frozen=True)
class TrainerConfig:
    device: str = "auto"  # auto | cpu | cuda
    precision: str = "fp32"  # fp32 | bf16
    seed: int = 42
    max_epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None

    num_workers: int = 0

    # Early stopping + checkpoint selection
    early_stopping: bool = True
    early_stopping_patience: int = 5
    monitor: str = "val_loss"  # val_loss | val_auc | val_rmse | ...
    monitor_mode: str = "min"  # min | max

    # Customization hooks
    loss_fn: str | None = None  # python callable ref: path/to.py:loss_fn


@dataclass(frozen=True)
class HPOConfig:
    enabled: bool = False
    # Minimal grid search (no new dependency) driven by config overrides.
    # Example: [ ["model.xgboost.max_depth", [4, 6, 8]], ["trainer.lr", [1e-3, 3e-4]] ]
    grid: list[tuple[str, list]] = field(default_factory=list)
    metric: str = "val_loss"
    mode: str = "min"  # min | max


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int = 6
    n_estimators: int = 500
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0


@dataclass(frozen=True)
class GRUConfig:
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


@dataclass(frozen=True)
class RNNConfig:
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    nonlinearity: str = "tanh"  # tanh | relu


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    pooling: str = "last"  # last | mean


@dataclass(frozen=True)
class ModelConfig:
    name: str  # xgboost | gru | rnn | transformer
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("logs")
    run_name: str = "run"
    save_preds: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    preprocess: PreprocessConfig
    task: TaskConfig
    split: SplitConfig
    model: ModelConfig
    models: list[ModelConfig] = field(default_factory=list)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    hpo_by_model: dict[str, HPOConfig] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
