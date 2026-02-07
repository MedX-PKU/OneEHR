from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = False
    # Only for binary tasks.
    method: str = "temperature"  # temperature | platt
    source: str = "val"  # val
    # Threshold selection on calibration set.
    threshold_strategy: str = "f1"  # f1
    # Whether to use calibrated probs for threshold selection + downstream outputs.
    use_calibrated: bool = True


@dataclass(frozen=True)
class DynamicTableConfig:
    path: Path | None = None
    # OneEHR consumes standardized CSV only.


@dataclass(frozen=True)
class StaticTableConfig:
    path: Path | None = None
    # OneEHR consumes standardized CSV only.


@dataclass(frozen=True)
class LabelTableConfig:
    path: Path | None = None
    # OneEHR consumes standardized CSV only.


@dataclass(frozen=True)
class DatasetConfig:
    """Three-table input spec.

    - `dynamic`: longitudinal event table (required)
    - `static`: patient-level table (optional)
    - `label`: label event table (optional; task-agnostic long format)

    OneEHR aims to keep dataset-specific raw formats out of the framework. Users
    should convert raw datasets into these standard tables outside OneEHR.
    """

    dynamic: DynamicTableConfig
    static: StaticTableConfig | None = None
    label: LabelTableConfig | None = None


@dataclass(frozen=True)
class DatasetsConfig:
    train: DatasetConfig
    test: DatasetConfig | None = None


@dataclass(frozen=True)
class PreprocessConfig:
    bin_size: str = "1d"  # e.g. 1h, 6h, 1d
    numeric_strategy: str = "mean"  # mean | last
    categorical_strategy: str = "onehot"  # onehot | count
    code_selection: str = "frequency"  # frequency | all | list | importance
    top_k_codes: int | None = 500
    min_code_count: int = 1
    code_list: list[str] = field(default_factory=list)
    importance_file: Path | None = None
    importance_code_col: str = "code"
    importance_value_col: str = "importance"

    # Post-merge preprocessing pipeline (applied after split on X).
    # Each step is a dict describing a built-in operator.
    # Example:
    #   pipeline = [
    #     {"op": "standardize", "cols": "num__*"},
    #     {"op": "impute", "strategy": "mean", "cols": "num__*"},
    #   ]
    pipeline: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class TaskConfig:
    kind: str  # binary | regression
    prediction_mode: str = "patient"  # patient | time


@dataclass(frozen=True)
class LabelsConfig:
    fn: str | None = None  # e.g. path/to/label_fn.py:build_labels
    bin_from_time_col: bool = True


@dataclass(frozen=True)
class SplitConfig:
    kind: str  # kfold | random | time
    seed: int = 42
    n_splits: int = 5
    val_size: float = 0.1
    test_size: float = 0.2
    time_boundary: str | None = None  # datetime string used for time split
    fold_index: int | None = None
    inner_kind: str | None = None  # for nested CV, e.g. time->kfold
    inner_n_splits: int | None = None


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
    final_refit: str = "train_val"  # train_only | train_val
    final_model_source: str = "refit"  # refit | best_split
    bootstrap_test: bool = False
    bootstrap_n: int = 200


@dataclass(frozen=True)
class HPOConfig:
    enabled: bool = False
    # Minimal grid search (no new dependency) driven by config overrides.
    # Example: [ ["model.xgboost.max_depth", [4, 6, 8]], ["trainer.lr", [1e-3, 3e-4]] ]
    grid: list[tuple[str, list]] = field(default_factory=list)
    metric: str = "val_loss"
    mode: str = "min"  # min | max
    scope: str = "single"  # single | per_split | cv_mean
    tune_split: str | None = None  # split name, e.g. fold0, split0, time0
    aggregate_metric: str | None = None  # metric key in metrics dict, e.g. auroc, rmse


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
class LSTMConfig:
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


@dataclass(frozen=True)
class MLPConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class TCNConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass(frozen=True)
class AdaCareConfig:
    hidden_dim: int = 128
    kernel_size: int = 2
    kernel_num: int = 64
    r_v: int = 4
    r_c: int = 4
    dropout: float = 0.5


@dataclass(frozen=True)
class StageNetConfig:
    hidden_dim: int = 384
    conv_size: int = 10
    levels: int = 3
    dropconnect: float = 0.3
    dropout: float = 0.3
    dropres: float = 0.3


@dataclass(frozen=True)
class RETAINConfig:
    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class ConCareConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1


@dataclass(frozen=True)
class GRASPConfig:
    hidden_dim: int = 128
    cluster_num: int = 12
    dropout: float = 0.1


@dataclass(frozen=True)
class MCGRUConfig:
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


@dataclass(frozen=True)
class AgentConfig:
    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class CatBoostConfig:
    depth: int = 6
    n_estimators: int = 500
    learning_rate: float = 0.05


@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 500
    max_depth: int | None = None


@dataclass(frozen=True)
class DTConfig:
    max_depth: int | None = None


@dataclass(frozen=True)
class GBDTConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 3


@dataclass(frozen=True)
class ModelConfig:
    name: str  # xgboost | catboost | rf | dt | gbdt | gru | rnn | lstm | mlp | tcn | transformer | ...
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    dt: DTConfig = field(default_factory=DTConfig)
    gbdt: GBDTConfig = field(default_factory=GBDTConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    adacare: AdaCareConfig = field(default_factory=AdaCareConfig)
    stagenet: StageNetConfig = field(default_factory=StageNetConfig)
    retain: RETAINConfig = field(default_factory=RETAINConfig)
    concare: ConCareConfig = field(default_factory=ConCareConfig)
    grasp: GRASPConfig = field(default_factory=GRASPConfig)
    mcgru: MCGRUConfig = field(default_factory=MCGRUConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


@dataclass(frozen=True)
class OutputConfig:
    root: Path = Path("logs")
    run_name: str = "run"
    save_preds: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    task: TaskConfig
    split: SplitConfig
    model: ModelConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    datasets: DatasetsConfig | None = None
    models: list[ModelConfig] = field(default_factory=list)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    hpo_by_model: dict[str, HPOConfig] = field(default_factory=dict)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
