# Configuration Reference

OneEHR experiments are driven by a single TOML config file. This page documents the current public config surface for the eval-first workflow.

See `examples/experiment.toml` for a complete working example.

---

## `[dataset]`

File paths for the three-table input spec. At minimum, `dynamic` is required.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamic` | `str` | required | Path to dynamic event CSV |
| `static` | `str` | `None` | Path to static patient CSV |
| `label` | `str` | `None` | Path to label event CSV |

```toml
[dataset]
dynamic = "data/dynamic.csv"
static = "data/static.csv"
label = "data/label.csv"
```

!!! tip
    If you use `labels.fn` to generate labels from events, you may omit `label`.

---

## `[datasets]`

For external test evaluation. Wraps `[dataset]` as `train` and adds a separate `test` dataset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train.dynamic` | `str` | required | Training dynamic CSV |
| `train.static` | `str` | `None` | Training static CSV |
| `train.label` | `str` | `None` | Training label CSV |
| `test.dynamic` | `str` | `None` | External-test dynamic CSV |
| `test.static` | `str` | `None` | External-test static CSV |
| `test.label` | `str` | `None` | External-test label CSV |

---

## `[preprocess]`

Controls how irregular events are binned and features are built.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bin_size` | `str` | `"1d"` | Time bin width such as `"1h"`, `"6h"`, or `"1d"` |
| `numeric_strategy` | `str` | `"mean"` | Aggregation for numeric values: `mean` or `last` |
| `categorical_strategy` | `str` | `"onehot"` | Encoding for categorical values: `onehot` or `count` |
| `code_selection` | `str` | `"frequency"` | Code vocabulary strategy: `frequency`, `all`, `list`, or `importance` |
| `top_k_codes` | `int` | `500` | Number of top codes for `frequency` or `importance` selection |
| `min_code_count` | `int` | `1` | Minimum occurrences for a code to be included |
| `code_list` | `list[str]` | `[]` | Explicit code list when `code_selection = "list"` |
| `importance_file` | `str` | `None` | Path to importance CSV when `code_selection = "importance"` |
| `importance_code_col` | `str` | `"code"` | Column name for code in importance file |
| `importance_value_col` | `str` | `"importance"` | Column name for importance score |
| `pipeline` | `list[dict]` | `[]` | Post-split preprocessing pipeline |

### Pipeline operators

The `pipeline` is a list of operator dicts applied in order after the train/val/test split. Each operator is fit on the training set only.

| Operator | Parameters | Description |
|----------|-----------|-------------|
| `standardize` | `cols` | Z-score normalization per column |
| `impute` | `strategy`, `cols` | Fill NaN values with `mean`, `median`, `mode`, or `constant` |
| `forward_fill` | `cols` | Within-patient temporal forward fill with fallback imputation |
| `clip` | `cols`, `lower`, `upper` | Hard clip to bounds |
| `winsorize` | `cols`, `lower_q`, `upper_q` | Quantile-based clipping fit on train |

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
| `kind` | `str` | required | Task type: `binary` or `regression` |
| `prediction_mode` | `str` | `"patient"` | Prediction granularity: `patient` or `time` |

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

See [Core Workflows](../guide/core-workflows.md#label-functions) for the callable contract.

---

## `[split]`

Patient-level group split configuration. All strategies guarantee no patient appears in multiple splits.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kind` | `str` | required | Split strategy: `kfold`, `random`, or `time` |
| `seed` | `int` | `42` | Random seed |
| `n_splits` | `int` | `5` | Number of folds for `kfold` |
| `val_size` | `float` | `0.1` | Validation fraction |
| `test_size` | `float` | `0.2` | Test fraction for `random` |
| `time_boundary` | `str` | `None` | Datetime boundary for `time` split |
| `fold_index` | `int` | `None` | Select a single fold from k-fold |
| `inner_kind` | `str` | `None` | Nested CV strategy for `time` or `random` |
| `inner_n_splits` | `int` | `None` | Number of inner folds for nested CV |

```toml
[split]
kind = "kfold"
n_splits = 5
seed = 42
val_size = 0.2
```

---

## `[model]` / `[[models]]`

Model selection and per-model hyperparameters. Use `[[models]]` to train multiple models in one experiment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Model name from the [Models](models.md) reference |

Each model name has a corresponding sub-table for model-specific parameters such as `[model.xgboost]`.

```toml
[[models]]
name = "gru"

[[models]]
name = "xgboost"
```

!!! note
    `[[models]]` is required for `oneehr train`. It is also the source of `kind = "trained_model"` entries in `[[eval.systems]]`.

---

## `[trainer]`

Training loop configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"auto"` | Device: `auto`, `cpu`, or `cuda` |
| `precision` | `str` | `"fp32"` | Precision: `fp32` or `bf16` |
| `seed` | `int` | `42` | Random seed |
| `max_epochs` | `int` | `30` | Maximum training epochs |
| `batch_size` | `int` | `64` | Batch size |
| `lr` | `float` | `1e-3` | Learning rate |
| `weight_decay` | `float` | `0.0` | AdamW weight decay |
| `grad_clip_norm` | `float` | `None` | Gradient clipping max norm |
| `num_workers` | `int` | `0` | DataLoader workers |
| `early_stopping` | `bool` | `true` | Enable early stopping |
| `early_stopping_patience` | `int` | `5` | Epochs without improvement before stopping |
| `monitor` | `str` | `"val_loss"` | Metric to monitor |
| `monitor_mode` | `str` | `"min"` | Optimization direction: `min` or `max` |
| `loss_fn` | `str` | `None` | Custom loss factory |
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
| `grid` | `list[list]` | `[]` | Grid axes such as `[["trainer.lr", [1e-3, 3e-4]]]` |
| `metric` | `str` | `"val_loss"` | Metric to optimize |
| `mode` | `str` | `"min"` | Optimization direction: `min` or `max` |
| `scope` | `str` | `"single"` | Selection scope: `single`, `per_split`, or `cv_mean` |
| `tune_split` | `str` | `None` | Split name to tune on |
| `aggregate_metric` | `str` | `None` | Metric key for `cv_mean` |

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

```toml
[calibration]
enabled = true
method = "temperature"
threshold_strategy = "f1"
```

---

## `[analysis]`

Controls the default behavior of `oneehr analyze`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_modules` | `list[str]` | see below | Modules to run when `--module` is not specified |
| `top_k` | `int` | `20` | Top-k rows/features retained in profile/drift tables |
| `stratify_by` | `list[str]` | `[]` | Static columns used for subgroup metrics in `prediction_audit` |
| `case_limit` | `int` | `50` | Max rows saved per case-level audit slice |
| `save_plot_specs` | `bool` | `true` | Save plot specs under `analysis/<module>/plots/` |
| `shap_max_samples` | `int` | `500` | Max rows used when SHAP is attempted in `interpretability` |

Common public module names:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `test_audit`
- `temporal_analysis`
- `interpretability`

```toml
[analysis]
default_modules = [
  "dataset_profile",
  "cohort_analysis",
  "prediction_audit",
  "test_audit",
  "temporal_analysis",
  "interpretability",
]
top_k = 20
stratify_by = []
case_limit = 50
save_plot_specs = true
shap_max_samples = 500
```

---

## `[eval]`

Controls frozen instance construction and report generation for the unified evaluation workflow.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instance_unit` | `str` | `"patient"` | Frozen instance unit: `patient` or `time` |
| `max_instances` | `int` | `None` | Optional cap on the number of eval instances |
| `seed` | `int` | `42` | Seed for bootstrap resampling and eval-level randomness |
| `include_static` | `bool` | `true` | Include static features in evidence bundles |
| `include_analysis_context` | `bool` | `false` | Attach analysis refs to evidence bundles |
| `max_events` | `int` | `200` | Maximum number of events stored per instance evidence bundle |
| `time_order` | `str` | `"asc"` | Event order in evidence bundles: `asc` or `desc` |
| `slice_by` | `list[str]` | `[]` | Reserved slice dimensions for downstream tooling |
| `primary_metric` | `str` | `None` | Override the report's primary ranking metric |
| `bootstrap_samples` | `int` | `200` | Number of bootstrap samples for pairwise comparisons |
| `save_evidence` | `bool` | `true` | Persist `eval/evidence/...` bundles |
| `save_traces` | `bool` | `true` | Persist `eval/traces/.../trace.parquet` |
| `text_render_template` | `str` | `"summary_v1"` | Template family used when rendering evidence text for framework prompts |

```toml
[eval]
instance_unit = "patient"
max_instances = 500
seed = 42
include_static = true
include_analysis_context = false
max_events = 200
time_order = "asc"
primary_metric = "auroc"
bootstrap_samples = 200
save_evidence = true
save_traces = true
text_render_template = "summary_v1"
```

Rules and constraints:

- `instance_unit` must be `patient` or `time`
- `max_events` and `bootstrap_samples` must be positive
- every `[[eval.systems]].sample_unit` must match `eval.instance_unit`

---

## `[[eval.backends]]`

One or more OpenAI-compatible chat-completion backends used by framework systems.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Logical backend name referenced by `backend_refs` |
| `provider` | `str` | `"openai_compatible"` | Provider type; v1 supports only `openai_compatible` |
| `base_url` | `str` | `"https://api.openai.com/v1"` | Base API URL |
| `model` | `str` | `""` | Provider model identifier |
| `api_key_env` | `str` | `"OPENAI_API_KEY"` | Environment variable containing the API key |
| `system_prompt` | `str` | `None` | Optional system prompt |
| `supports_json_schema` | `bool` | `true` | Send OpenAI-style JSON schema response formats when supported |
| `prompt_token_cost_per_1k` | `float` | `None` | Optional prompt token cost used for saved cost accounting |
| `completion_token_cost_per_1k` | `float` | `None` | Optional completion token cost used for saved cost accounting |
| `headers` | `table` | `{}` | Extra request headers |

```toml
[[eval.backends]]
name = "gpt4o-mini"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
supports_json_schema = true
prompt_token_cost_per_1k = 0.002
completion_token_cost_per_1k = 0.004
headers = {}
```

---

## `[[eval.systems]]`

Defines one scored system in the unified eval suite.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | System name used in `eval/predictions/`, `eval/traces/`, and reports |
| `kind` | `str` | `"framework"` | `framework` or `trained_model` |
| `framework_type` | `str` | `None` | Required for framework systems; see supported list below |
| `enabled` | `bool` | `true` | Whether the system participates in `eval run` |
| `sample_unit` | `str` | `"patient"` | Must match `eval.instance_unit` |
| `source_model` | `str` | `None` | Required for `kind = "trained_model"` |
| `backend_refs` | `list[str]` | `[]` | Backend names from `[[eval.backends]]` |
| `prompt_template` | `str` | `"summary_v1"` | Prompt template family for framework prompts |
| `max_samples` | `int` | `None` | Optional cap on rows scored by this system |
| `max_rounds` | `int` | `1` | Maximum dialogue/discussion rounds |
| `concurrency` | `int` | `1` | Concurrent framework requests |
| `max_retries` | `int` | `2` | Retry count for retryable failures |
| `timeout_seconds` | `float` | `60.0` | Per-request timeout |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `top_p` | `float` | `1.0` | Sampling top-p |
| `seed` | `int` | `None` | Optional request seed |
| `framework_params` | `table` | `{}` | Framework-specific optional settings |

Supported `framework_type` values:

- `single_llm`
- `healthcareagent`
- `reconcile`
- `mac`
- `medagent`
- `colacare`
- `mdagents`

```toml
[[eval.systems]]
name = "xgboost_ref"
kind = "trained_model"
sample_unit = "patient"
source_model = "xgboost"

[[eval.systems]]
name = "mdagents_eval"
kind = "framework"
framework_type = "mdagents"
sample_unit = "patient"
backend_refs = ["gpt4o-mini"]
max_rounds = 2
framework_params = { num_teams_advanced = 2, num_agents_per_team_advanced = 2 }
```

Rules and constraints:

- `kind = "trained_model"` systems must set `source_model` and must not set `framework_type` or `backend_refs`
- `kind = "framework"` systems must set `framework_type` and valid `backend_refs`
- system names must be unique

---

## `[[eval.suites]]`

Optional reporting suites used to define filtered comparisons and explicit paired deltas.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Suite name written into paired-comparison rows |
| `splits` | `list[str]` | `[]` | Optional split filter |
| `include_systems` | `list[str]` | `[]` | Optional system subset |
| `primary_metric` | `str` | `None` | Suite-specific primary metric |
| `secondary_metrics` | `list[str]` | `[]` | Optional additional metrics |
| `slice_by` | `list[str]` | `[]` | Reserved slice dimensions |
| `min_coverage` | `float` | `0.0` | Minimum required coverage in `[0, 1]` |
| `compare_pairs` | `list[tuple[str, str]]` | `[]` | Explicit paired comparisons |

```toml
[[eval.suites]]
name = "core"
primary_metric = "auroc"
include_systems = ["xgboost_ref", "single_llm_eval", "mdagents_eval"]
compare_pairs = [
  ["xgboost_ref", "single_llm_eval"],
  ["xgboost_ref", "mdagents_eval"],
]
```

---

## `[output]`

Run directory and artifact configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str` | `"logs"` | Root directory for all runs |
| `run_name` | `str` | `"run"` | Name of this experiment run |
| `save_preds` | `bool` | `true` | Save predictions as parquet files |

Artifacts are written to `{root}/{run_name}/`. See [Artifacts](artifacts.md) for the on-disk layout.

```toml
[output]
root = "logs"
run_name = "example"
save_preds = true
```
