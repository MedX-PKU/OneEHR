# Configuration Reference

OneEHR experiments are driven by a single TOML config file. This page documents the public configuration contract for preprocessing, modeling, testing, and analysis.

See `examples/tjh/experiment.toml` for a complete working example.

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
| `numeric_strategy` | `str` | `"mean"` | Aggregation for numeric values: `mean` or `last` |
| `categorical_strategy` | `str` | `"onehot"` | Encoding for categorical values: `onehot` or `count` |
| `code_selection` | `str` | `"frequency"` | Code vocabulary strategy: `frequency`, `all`, or `list` |
| `top_k_codes` | `int` | `100` | Number of top codes for `frequency` selection |

```toml
[preprocess]
bin_size = "1d"
numeric_strategy = "mean"
categorical_strategy = "onehot"
code_selection = "frequency"
top_k_codes = 100
```

---

## `[task]`

Defines the prediction task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kind` | `str` | `"binary"` | Task type: `binary` or `regression` |
| `prediction_mode` | `str` | `"patient"` | Prediction granularity: `patient` or `time` |

```toml
[task]
kind = "binary"
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
| `precision` | `str` | `"fp32"` | `fp32` or `bf16` |
| `early_stopping` | `bool` | `true` | Enable early stopping |
| `patience` | `int` | `5` | Epochs without improvement before stopping |

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
