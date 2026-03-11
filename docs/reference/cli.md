# CLI Reference

OneEHR provides a single `oneehr` CLI with six subcommands that map to the experiment pipeline: **preprocess**, **train**, **test**, **analyze**, **llm-preprocess**, **llm-predict**.

Run `uv run oneehr --help` for the authoritative list of flags.

---

## `oneehr preprocess`

Validate input data, bin irregular events into fixed time bins, build features, and materialize tabular views.

```
oneehr preprocess --config <toml> [--overview] [--overview-top-k-codes N]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--overview` | flag | `false` | Print a dataset overview JSON to stdout |
| `--overview-top-k-codes` | `int` | `20` | Number of top codes to show in overview |

**What it does:**

1. Loads config, reads dynamic/static/label CSV files
2. Bins events into fixed time bins (`preprocess.bin_size`)
3. Selects code vocabulary per `preprocess.code_selection`
4. Builds `num__*` and `cat__*` feature columns
5. Materializes patient or time tabular views
6. Runs `labels.fn` (if configured) and joins labels onto views
7. Writes `run_manifest.json` and all preprocess artifacts

**Example:**

```bash
uv run oneehr preprocess --config examples/experiment.toml
uv run oneehr preprocess --config examples/experiment.toml --overview
```

---

## `oneehr train`

Train one or more models with optional config-driven grid search (HPO) and evaluation.

```
oneehr train --config <toml> [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--force` | flag | `false` | Overwrite existing run directory |

**What it does:**

1. Loads config and precomputed artifacts from the run directory
2. Creates patient-level group splits
3. For each model in `[[models]]`:
    - Resolves per-model HPO config (`[hpo]` or `[hpo_models.<name>]`)
    - Runs grid search if enabled
    - For each split: trains, evaluates, optionally calibrates
    - Saves model checkpoints, metrics, and predictions
4. Writes `summary.json` and `paper_table.csv`

**Example:**

```bash
uv run oneehr train --config examples/experiment.toml
uv run oneehr train --config examples/experiment.toml --force
```

---

## `oneehr test`

Evaluate trained models on a test dataset.

```
oneehr test --config <toml> [--run-dir DIR] [--test-dataset PATH] [--force] [--out-dir DIR]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--test-dataset` | `str` | `None` | Path to test dataset config (overrides `datasets.test`) |
| `--force` | flag | `false` | Overwrite existing test results |
| `--out-dir` | `str` | `None` | Output directory for test results |

**What it does:**

1. Reads the run manifest from the training run
2. Loads the test dataset and re-bins using the train-time code vocabulary
3. Aligns feature columns to match the training schema
4. For each trained model/split: loads model, generates predictions, computes metrics
5. Writes per-model/split metrics and `test_summary.json`

**Example:**

```bash
uv run oneehr test --config examples/experiment.toml
uv run oneehr test --config examples/experiment.toml --run-dir logs/example --test-dataset path/to/test.toml
```

---

## `oneehr analyze`

Run feature importance analysis on trained models.

```
oneehr analyze --config <toml> [--run-dir DIR] [--method xgboost|shap|attention]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--method` | `str` | `None` | Analysis method: `xgboost`, `shap`, or `attention` |

**What it does:**

1. Loads the run manifest and reads the tabular view (features + labels)
2. For each trained model/split:
    - **xgboost**: native feature importance (gain-based)
    - **shap**: SHAP values via `shap.Explainer`
    - **attention**: attention-weighted feature magnitudes (DL models with attention)
3. Writes JSON results to `analysis/feature_importance_{model}_{split}_{method}.json`

!!! note "Method availability"
    - `xgboost` method works with XGBoost models only
    - `shap` works with any model but requires `shap` (included in dependencies)
    - `attention` works with DL models that expose attention weights
    - For tabular models, `--method` is auto-selected if omitted

**Example:**

```bash
uv run oneehr analyze --config examples/experiment.toml
uv run oneehr analyze --config examples/experiment.toml --method shap
```

---

## `oneehr llm-preprocess`

Materialize LLM evaluation instances from the run directory. This command uses the existing patient-level split contract and writes prompt-ready instance tables under `llm/instances/`.

```
oneehr llm-preprocess --config <toml> [--run-dir DIR] [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--force` | flag | `false` | Overwrite existing LLM instance artifacts |

**What it does:**

1. Loads the existing `run_manifest.json` produced by `oneehr preprocess`
2. Reuses saved patient-group splits if present; otherwise materializes them from the current split config
3. Builds patient-level or time-level LLM instances from held-out test patients only
4. Writes `llm/instances/patient_instances.parquet` or `llm/instances/time_instances.parquet`
5. Writes `llm/instances/summary.json`

**Example:**

```bash
uv run oneehr llm-preprocess --config examples/experiment.toml
uv run oneehr llm-preprocess --config examples/experiment.toml --run-dir logs/example --force
```

---

## `oneehr llm-predict`

Render structured EHR prompts, call OpenAI-compatible chat completions, parse strict JSON outputs, and score predictions.

```
oneehr llm-predict --config <toml> [--run-dir DIR] [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--force` | flag | `false` | Overwrite existing LLM prediction artifacts |

**What it does:**

1. Loads `llm/instances/*.parquet` (materializing them first if missing)
2. Renders the built-in `summary_v1` prompt for each instance
3. Calls each `[[llm_models]]` entry through an OpenAI-compatible `chat/completions` endpoint
4. Parses strict JSON responses for `binary` or `regression` tasks
5. Writes prompts, raw responses, parsed outputs, predictions, failures, per-split metrics, and `llm/summary.json`

**Example:**

```bash
uv run oneehr llm-predict --config examples/experiment.toml
uv run oneehr llm-predict --config examples/experiment.toml --run-dir logs/example --force
```
