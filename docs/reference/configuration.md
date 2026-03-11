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
| `inner_kind` | `str` | `None` | Nested CV strategy for `time` or `random` split (e.g. `"kfold"`) |
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

!!! note
    The new LLM workflow uses `[llm]` and `[[llm_models]]` instead of the training model registry. LLM-only configs are allowed: for `preprocess`, `llm-preprocess`, and `llm-predict`, you do not need to define `[model]` or `[[models]]`.

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
| `repeat` | `int` | `1` | Number of training runs per split with different seeds |

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

## `[llm]`

Top-level configuration for the LLM inference/evaluation workflow.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable the LLM workflow |
| `sample_unit` | `str` | `"patient"` | LLM evaluation unit: `patient` or `time` |
| `prompt_template` | `str` | `"summary_v1"` | Prompt renderer template (currently only `summary_v1`) |
| `json_schema_version` | `int` | `1` | Output schema version (currently only `1`) |
| `max_samples` | `int` | `None` | Optional cap on the number of materialized LLM instances |
| `save_prompts` | `bool` | `true` | Save rendered prompts to `llm/prompts/` |
| `save_responses` | `bool` | `true` | Save raw responses to `llm/responses/` |
| `save_parsed` | `bool` | `true` | Save parsed outputs to `llm/parsed/` |
| `concurrency` | `int` | `1` | Number of concurrent LLM requests |
| `max_retries` | `int` | `2` | Retry count for retryable HTTP/network failures |
| `timeout_seconds` | `float` | `60.0` | Per-request timeout |
| `temperature` | `float` | `0.0` | Chat completion temperature |
| `top_p` | `float` | `1.0` | Chat completion top-p |
| `seed` | `int` | `None` | Optional request seed (provider support varies) |

Rules and constraints:

- `llm.sample_unit` must match `task.prediction_mode`
- Supported task kinds are `binary` and `regression`
- Supported providers are OpenAI-compatible `chat/completions` only
- `llm.prompt.include_labels_context` is intentionally disabled in v1 to prevent leakage

```toml
[llm]
enabled = true
sample_unit = "patient"
prompt_template = "summary_v1"
json_schema_version = 1
save_prompts = true
save_responses = true
save_parsed = true
concurrency = 1
max_retries = 2
timeout_seconds = 60.0
temperature = 0.0
top_p = 1.0
```

---

## `[llm.prompt]`

Prompt construction options for `summary_v1`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_static` | `bool` | `true` | Include static patient features when available |
| `include_labels_context` | `bool` | `false` | Reserved; currently must remain `false` |
| `history_window` | `str` | `None` | Optional retrospective window (e.g. `"7d"`, `"48h"`) |
| `max_events` | `int` | `200` | Maximum number of raw events rendered into the prompt |
| `time_order` | `str` | `"asc"` | Event order in the rendered timeline: `asc` or `desc` |
| `sections` | `list[str]` | see below | Prompt sections to include |

Default sections:

- `patient_profile`
- `event_timeline`
- `code_summary`
- `prediction_task`
- `output_schema`

```toml
[llm.prompt]
include_static = true
history_window = "30d"
max_events = 150
time_order = "asc"
sections = ["patient_profile", "event_timeline", "prediction_task", "output_schema"]
```

---

## `[llm.output]`

Controls optional non-metric fields requested from the LLM.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_explanation` | `bool` | `true` | Ask the model for an explanation field |
| `include_confidence` | `bool` | `false` | Ask the model for a confidence field in `[0, 1]` |

Binary tasks always expect JSON containing `label` and preferably `probability`. If `probability` is missing but `label` parses correctly, OneEHR falls back to `0.0` or `1.0`.

Regression tasks expect JSON containing `value`.

```toml
[llm.output]
include_explanation = true
include_confidence = false
```

---

## `[[llm_models]]`

One or more OpenAI-compatible chat completion backends to evaluate on the same instances.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Logical name used in `llm/` artifact paths |
| `provider` | `str` | `"openai_compatible"` | Provider type (currently only `openai_compatible`) |
| `base_url` | `str` | `"https://api.openai.com/v1"` | Base API URL; `llm-predict` appends `/chat/completions` |
| `model` | `str` | *required* | Provider model identifier |
| `api_key_env` | `str` | `"OPENAI_API_KEY"` | Environment variable containing the API key |
| `system_prompt` | `str` | `None` | Optional system prompt |
| `supports_json_schema` | `bool` | `true` | Send OpenAI-style `response_format = {type = "json_schema"}` |
| `headers` | `table` | `{}` | Extra request headers for compatible vendors |

```toml
[[llm_models]]
name = "gpt4o-mini"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
supports_json_schema = true

[[llm_models]]
name = "local-vllm"
provider = "openai_compatible"
base_url = "http://127.0.0.1:8000/v1"
model = "meta-llama/Llama-3.1-8B-Instruct"
api_key_env = "VLLM_API_KEY"
supports_json_schema = false
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
