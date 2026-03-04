# Configuration Reference

OneEHR experiments are driven by a single TOML config file. This page documents every section and parameter.

See `examples/experiment.toml` for a complete working example.

---

## `[dataset]`

File paths for the three-table input spec. At minimum, `dynamic` is required.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamic` | `str` | *required* | Path to dynamic event CSV |
| `static` | `str` | `None` | Path to static patient CSV |
| `label` | `str` | `None` | Path to label event CSV |

```toml
[dataset]
dynamic = "data/dynamic.csv"
static = "data/static.csv"     # optional
label = "data/label.csv"       # optional
```

!!! tip
    If you use `labels.fn` to generate labels from events, you may omit `label`.

---

## `[datasets]`

For external test evaluation. Wraps `[dataset]` as `train` and adds a separate `test` dataset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train.dynamic` | `str` | | Training dynamic CSV |
| `train.static` | `str` | `None` | Training static CSV |
| `train.label` | `str` | `None` | Training label CSV |
| `test.dynamic` | `str` | `None` | Test dynamic CSV |
| `test.static` | `str` | `None` | Test static CSV |
| `test.label` | `str` | `None` | Test label CSV |

---

## `[preprocess]`

Controls how irregular events are binned and features are built.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bin_size` | `str` | `"1d"` | Time bin width (e.g. `"1h"`, `"6h"`, `"1d"`) |
| `numeric_strategy` | `str` | `"mean"` | Aggregation for numeric values: `mean` or `last` |
| `categorical_strategy` | `str` | `"onehot"` | Encoding for categorical values: `onehot` or `count` |
| `code_selection` | `str` | `"frequency"` | Code vocabulary strategy: `frequency`, `all`, `list`, or `importance` |
| `top_k_codes` | `int` | `500` | Number of top codes for `frequency` or `importance` selection |
| `min_code_count` | `int` | `1` | Minimum occurrences for a code to be included |
| `code_list` | `list[str]` | `[]` | Explicit code list (when `code_selection = "list"`) |
| `importance_file` | `str` | `None` | Path to importance CSV (when `code_selection = "importance"`) |
| `importance_code_col` | `str` | `"code"` | Column name for code in importance file |
| `importance_value_col` | `str` | `"importance"` | Column name for importance score |
| `pipeline` | `list[dict]` | `[]` | Post-merge preprocessing pipeline (fit on train split) |

### Pipeline operators

The `pipeline` is a list of operator dicts applied in order after the train/val/test split. Each operator is fit on the training set only.

| Operator | Parameters | Description |
|----------|-----------|-------------|
| `standardize` | `cols` | Z-score normalization per column |
| `impute` | `strategy`, `cols` | Fill NaN values (`mean`, `median`, `mode`, or `constant`) |
| `forward_fill` | `cols` | Within-patient temporal forward fill with fallback imputation |
| `clip` | `cols`, `lower`, `upper` | Hard clip to bounds |
| `winsorize` | `cols`, `lower_q`, `upper_q` | Quantile-based clipping (fit on train) |

The `cols` parameter supports glob patterns (e.g. `"num__*"`, `"cat__*"`).

```toml
[preprocess]
bin_size = "1d"
numeric_strategy = "mean"
categorical_strategy = "onehot"
code_selection = "frequency"
top_k_codes = 100
pipeline = [
  { op = "standardize", cols = "num__*" },
  { op = "impute", strategy = "mean", cols = "num__*" },
]
```

---

## `[task]`

Defines the prediction task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kind` | `str` | *required* | Task type: `binary` or `regression` |
| `prediction_mode` | `str` | `"patient"` | Prediction granularity: `patient` (N-1) or `time` (N-N) |

```toml
[task]
kind = "binary"
prediction_mode = "patient"
```

---

## `[labels]`

Optional label generation from a Python function.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `str` | `None` | Python callable reference: `"path/to/file.py:function_name"` |
| `bin_from_time_col` | `bool` | `true` | Floor `label_time` to bin boundaries using `preprocess.bin_size` |

```toml
[labels]
fn = "examples/label_fn.py:build_labels"
bin_from_time_col = true
```

---

## `[split]`

Patient-level group split configuration. All strategies guarantee no patient appears in multiple splits.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kind` | `str` | *required* | Split strategy: `kfold`, `random`, or `time` |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `n_splits` | `int` | `5` | Number of folds (for `kfold`) |
| `val_size` | `float` | `0.1` | Validation fraction |
| `test_size` | `float` | `0.2` | Test fraction (for `random`) |
| `time_boundary` | `str` | `None` | Datetime string for `time` split (e.g. `"2012-01-01"`) |
| `fold_index` | `int` | `None` | Select a single fold from k-fold (0-indexed) |
| `inner_kind` | `str` | `None` | Nested CV strategy for `time` split (e.g. `"kfold"`) |
| `inner_n_splits` | `int` | `None` | Number of inner folds for nested CV |

=== "K-fold CV"

    ```toml
    [split]
    kind = "kfold"
    n_splits = 5
    seed = 42
    val_size = 0.2
    ```

=== "Random split"

    ```toml
    [split]
    kind = "random"
    seed = 42
    val_size = 0.1
    test_size = 0.2
    ```

=== "Time split"

    ```toml
    [split]
    kind = "time"
    time_boundary = "2012-01-01"
    ```

=== "Time + nested CV"

    ```toml
    [split]
    kind = "time"
    time_boundary = "2012-01-01"
    inner_kind = "kfold"
    inner_n_splits = 5
    ```

---

## `[model]` / `[[models]]`

Model selection and per-model hyperparameters. Use `[[models]]` (array of tables) to train multiple models in one experiment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Model name (see [Models](models.md)) |

Each model name has a corresponding sub-table for model-specific parameters (e.g. `[model.xgboost]`). See [Models](models.md) for all parameters.

```toml
[[models]]
name = "gru"

[[models]]
name = "xgboost"
```

---

## `[trainer]`

Training loop configuration (applies to all models, especially deep learning).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"auto"` | Device: `auto`, `cpu`, or `cuda` |
| `precision` | `str` | `"fp32"` | Floating point precision: `fp32` or `bf16` |
| `seed` | `int` | `42` | Random seed |
| `max_epochs` | `int` | `30` | Maximum training epochs |
| `batch_size` | `int` | `64` | Batch size |
| `lr` | `float` | `1e-3` | Learning rate |
| `weight_decay` | `float` | `0.0` | AdamW weight decay |
| `grad_clip_norm` | `float` | `None` | Gradient clipping max norm |
| `num_workers` | `int` | `0` | DataLoader workers |
| `early_stopping` | `bool` | `true` | Enable early stopping |
| `early_stopping_patience` | `int` | `5` | Epochs without improvement before stopping |
| `monitor` | `str` | `"val_loss"` | Metric to monitor for early stopping |
| `monitor_mode` | `str` | `"min"` | Optimization direction: `min` or `max` |
| `loss_fn` | `str` | `None` | Custom loss factory: `"path/to.py:loss_fn"` |
| `final_refit` | `str` | `"train_val"` | Final model refit: `train_only` or `train_val` |
| `final_model_source` | `str` | `"refit"` | Source for final model: `refit` or `best_split` |
| `bootstrap_test` | `bool` | `false` | Run bootstrap test evaluation |
| `bootstrap_n` | `int` | `200` | Number of bootstrap samples |

```toml
[trainer]
device = "auto"
max_epochs = 30
batch_size = 64
lr = 1e-3
early_stopping = true
early_stopping_patience = 5
monitor = "val_loss"
final_refit = "train_val"
```

---

## `[hpo]`

Config-driven grid search. HPO runs inside `oneehr train` when enabled.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable grid search |
| `grid` | `list[list]` | `[]` | Grid axes: `[["dotted.key", [val1, val2, ...]], ...]` |
| `metric` | `str` | `"val_loss"` | Metric to optimize |
| `mode` | `str` | `"min"` | Optimization direction: `min` or `max` |
| `scope` | `str` | `"single"` | Selection scope: `single`, `per_split`, or `cv_mean` |
| `tune_split` | `str` | `None` | Split name to tune on (e.g. `"fold0"`) |
| `aggregate_metric` | `str` | `None` | Metric key for `cv_mean` (e.g. `"auroc"`, `"rmse"`) |

See [HPO Guide](../guide/hpo.md) for usage patterns.

```toml
[hpo]
enabled = true
metric = "val_loss"
mode = "min"
scope = "single"
grid = [
  ["trainer.lr", [1e-3, 3e-4]],
  ["trainer.batch_size", [16, 32]],
]
```

---

## `[hpo_models.<name>]`

Per-model HPO overrides. Same parameters as `[hpo]`. The model name must match a `[[models]]` entry.

```toml
[hpo_models.xgboost]
enabled = true
metric = "val_auroc"
mode = "max"
grid = [
  ["model.xgboost.max_depth", [4, 6, 8]],
  ["model.xgboost.n_estimators", [200, 500]],
]
```

---

## `[calibration]`

Post-hoc probability calibration for binary classification tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable calibration |
| `method` | `str` | `"temperature"` | Calibration method: `temperature` or `platt` |
| `source` | `str` | `"val"` | Calibration data source |
| `threshold_strategy` | `str` | `"f1"` | Threshold selection strategy |
| `use_calibrated` | `bool` | `true` | Use calibrated probabilities for threshold selection |

See [Calibration Guide](../guide/calibration.md) for details.

```toml
[calibration]
enabled = true
method = "temperature"
threshold_strategy = "f1"
```

---

## `[output]`

Run directory and artifact configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str` | `"logs"` | Root directory for all runs |
| `run_name` | `str` | `"run"` | Name of this experiment run |
| `save_preds` | `bool` | `true` | Save predictions as parquet files |

Artifacts are written to `{root}/{run_name}/`. See [Artifacts](artifacts.md) for the full directory layout.

```toml
[output]
root = "logs"
run_name = "example"
save_preds = true
```
