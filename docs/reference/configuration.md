# Configuration Reference

OneEHR experiments are driven by a single TOML config file. This page documents the public configuration contract for preprocessing, modeling, testing, and analysis.

See `examples/tjh/mortality_patient.toml` for a complete working example.

---

## `[dataset]`

File paths for the three-table input spec. At minimum, `dynamic` is required.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamic` | `str` | `None` | Path to dynamic event CSV |
| `static` | `str` | `None` | Path to static patient CSV |
| `label` | `str` | `None` | Path to label event CSV |

```toml
[dataset]
dynamic = "data/dynamic.csv"
static = "data/static.csv"
label = "data/label.csv"
```

---

## `[preprocess]`

Controls how irregular events are binned and features are built.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bin_size` | `str` | `"1d"` | Time bin width, for example `"1h"`, `"6h"`, or `"1d"` |
| `numeric_strategy` | `str` | `"mean"` | Aggregation for numeric values: `mean`, `last`, `median`, `min`, `max`, `std`, or `count` |
| `categorical_strategy` | `str` | `"onehot"` | Encoding for categorical values: `onehot` or `count` |
| `code_selection` | `str` | `"frequency"` | Code vocabulary strategy: `frequency`, `all`, or `list` |
| `top_k_codes` | `int` | `100` | Number of top codes for `frequency` selection |
| `min_code_count` | `int` | `1` | Minimum event count for a code to be included in the vocabulary |
| `max_seq_length` | `int` | `None` | Truncate sequences to most recent N time bins |
| `min_events_per_patient` | `int` | `1` | Exclude patients with fewer events |
| `pipeline` | `list[dict]` | `[]` | Ordered list of preprocessing ops applied after binning (see below) |

```toml
[preprocess]
bin_size = "1d"
numeric_strategy = "mean"
categorical_strategy = "onehot"
code_selection = "frequency"
top_k_codes = 100
min_code_count = 1
```

### Preprocessing Pipeline

The `pipeline` field defines an ordered sequence of preprocessing operations fitted on the train split and applied to all splits. When `pipeline` is empty (the default), numeric features are filled with 0 at train/test time as a safety net.

Each step is a TOML table with an `op` key and operation-specific parameters. The `cols` parameter supports glob patterns (`"num__*"`, `"cat__*"`), explicit lists, or `null` (all columns).

Supported operations:

| Op | Description | Key params |
|----|-------------|------------|
| `impute` | Fill NaN with a statistic | `strategy` (`mean`, `median`, `mode`, `constant`), `value` |
| `forward_fill` | LOCF within patient + fallback | `group_key`, `order_key`, `fallback.strategy` |
| `standardize` | Z-score normalization | _(none)_ |
| `zscore_filter` | Replace outliers beyond threshold with NaN | `threshold` (default `3.0`) |
| `normalize_label` | Z-score the label column (regression) | `col` (default `"label"`) |
| `winsorize` | Quantile-based outlier clipping | `lower_q`, `upper_q` |
| `clip` | Hard value clipping | `lower`, `upper` |
| `knn_impute` | KNN imputation | `n_neighbors` |
| `iterative_impute` | MICE imputation | `max_iter` |
| `robust_scale` | Median/IQR scaling | _(none)_ |
| `quantile_norm` | Quantile normalization | `output_distribution`, `n_quantiles` |

Example: LOCF with mean fallback (recommended for time-series EHR):

```toml
[preprocess]
pipeline = [
  { op = "forward_fill", cols = "num__*", group_key = "patient_id", order_key = "bin_time", fallback = { strategy = "mean" } },
]
```

Example: Outlier handling + imputation + normalization:

```toml
[preprocess]
pipeline = [
  { op = "zscore_filter", cols = "num__*", threshold = 3.0 },
  { op = "impute", cols = "num__*", strategy = "median" },
  { op = "standardize", cols = "num__*" },
]
```

---

## `[task]`

Defines the prediction task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kind` | `str` | `"binary"` | Task type: `binary`, `regression`, `multiclass`, `survival`, or `multilabel` |
| `prediction_mode` | `str` | `"patient"` | Prediction granularity: `patient` or `time` |
| `num_classes` | `int` | `None` | Number of classes (required when `kind = "multiclass"`) |
| `loss` | `str` | `"default"` | Loss function: `default` or `focal` |
| `focal_gamma` | `float` | `2.0` | Gamma for focal loss |

```toml
# Binary classification
[task]
kind = "binary"
prediction_mode = "patient"
```

```toml
# Multiclass classification
[task]
kind = "multiclass"
prediction_mode = "patient"
num_classes = 5
```

```toml
# Survival analysis (time-to-event with censoring)
[task]
kind = "survival"
prediction_mode = "patient"
```

```toml
# Multi-label classification (e.g., ICD coding)
[task]
kind = "multilabel"
prediction_mode = "patient"
```

---

## `[split]`

Patient-level group split configuration. All strategies guarantee no patient appears in multiple splits.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kind` | `str` | `"random"` | Split strategy: `random` or `time` |
| `seed` | `int` | `42` | Random seed |
| `val_size` | `float` | `0.1` | Validation fraction |
| `test_size` | `float` | `0.2` | Test fraction for `random` |
| `time_boundary` | `str` | `None` | Datetime string for `time` splits |

```toml
[split]
kind = "random"
seed = 42
val_size = 0.1
test_size = 0.2
```

```toml
[split]
kind = "time"
time_boundary = "2012-01-01"
```

---

## `[[models]]`

Model selection and per-model hyperparameters. Use `[[models]]` to train multiple models in one experiment. Each entry has a `name` and an optional `params` dict for model-specific hyperparameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"xgboost"` | Model name, see [Models](models.md) |
| `params` | `dict` | `{}` | Model-specific hyperparameters |

```toml
[[models]]
name = "xgboost"
[models.params]
n_estimators = 100
max_depth = 4
learning_rate = 0.1

[[models]]
name = "gru"
[models.params]
hidden_dim = 64
num_layers = 1
```

---

## `[trainer]`

Training loop configuration for deep learning models.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"auto"` | `auto`, `cpu`, or `cuda` |
| `seed` | `int` | `42` | Random seed |
| `max_epochs` | `int` | `30` | Maximum training epochs |
| `batch_size` | `int` | `64` | Batch size |
| `lr` | `float` | `1e-3` | Learning rate |
| `weight_decay` | `float` | `0.0` | AdamW weight decay |
| `grad_clip` | `float` | `1.0` | Gradient clipping max norm |
| `num_workers` | `int` | `0` | DataLoader workers |
| `precision` | `str` | `"fp32"` | `fp32`, `fp16`, or `bf16` |
| `scheduler` | `str` | `"none"` | LR scheduler: `none`, `cosine`, `step`, or `plateau` |
| `scheduler_params` | `dict` | `{}` | Scheduler-specific params (e.g., `T_max`, `step_size`, `gamma`) |
| `class_weight` | `str` | `"none"` | Class weighting: `none` or `balanced` |
| `early_stopping` | `bool` | `true` | Enable early stopping |
| `patience` | `int` | `5` | Epochs without improvement before stopping |
| `monitor` | `str` | `"val_loss"` | Metric for early stopping: `val_loss`, `val_auroc`, `val_auprc`, `val_rmse`, `val_mae` |
| `monitor_mode` | `str` | `"min"` | `min` (lower is better) or `max` (higher is better) |

```toml
[trainer]
device = "auto"
seed = 42
max_epochs = 30
batch_size = 64
lr = 1e-3
early_stopping = true
patience = 5
```

Monitor AUROC instead of loss for binary tasks:

```toml
[trainer]
monitor = "val_auroc"
monitor_mode = "max"
early_stopping = true
patience = 10
```

The trainer tracks a per-epoch history of `train_loss`, `val_loss`, and the monitored metric (if not `val_loss`). This history is saved in `meta.json` under `train_metrics.history`.

---

## `[[systems]]`

LLM system definitions for cross-system comparison via `oneehr test`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Unique system name |
| `kind` | `str` | `"llm"` | System kind: `llm` or `agent` |
| `framework` | `str` | `"single_llm"` | Framework type |
| `backend` | `str` | `"openai"` | Backend provider |
| `model` | `str` | `"gpt-4o"` | Provider model identifier |
| `api_key_env` | `str` | `"OPENAI_API_KEY"` | Environment variable containing the API key |
| `params` | `dict` | `{}` | System-specific parameters |

```toml
[[systems]]
name = "gpt4o_eval"
kind = "llm"
framework = "single_llm"
backend = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
```

---

## `[output]`

Run directory configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str` | `"runs"` | Root directory for all runs |
| `run_name` | `str` | `"exp001"` | Name of this experiment run |

Artifacts are written to `{root}/{run_name}/`. See [Artifacts](artifacts.md) for the full directory layout.

```toml
[output]
root = "runs"
run_name = "tjh"
```
