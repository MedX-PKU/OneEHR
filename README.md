# OneEHR

OneEHR is an end-to-end EHR predictive modeling + analysis toolkit in Python, designed around:

- **Doctor-friendly input**: start from a single long-form event table (CSV/Excel).
- **Leakage prevention by default**: all data splits are patient-level group splits.
- **TOML-first experiments**: most behavior is configured via one `experiment.toml`.
- **CLI-first workflow**: preprocess → train (with optional grid search) → test → analyze.

This README is the user-facing guide. If you are using agents to modify the repo, read `AGENTS.md` too.

## Install (uv-only)

Python is pinned in `.python-version` (Python 3.12). Use `uv`:

```bash
uv venv .venv --python 3.12
uv pip install -e .
```

Verify:

```bash
uv run oneehr --help
```

## Quickstart (5 minutes)

Run the example end-to-end:

```bash
uv run oneehr preprocess --config examples/experiment.toml
uv run oneehr train --config examples/experiment.toml
uv run oneehr analyze --config examples/experiment.toml
```

Artifacts are written under `output.root/output.run_name` (defaults in config; the example uses `logs/example/`).

## The Workflow (Conceptual)

OneEHR is organized around an explicit pipeline:

1. **Preprocess**: validate and bin irregular events into fixed time bins; build features; materialize tabular “views”.
2. **Train**: fit one or more models; optionally run config-driven grid search (HPO) per model.
3. **Test**: evaluate a trained run on external test data (dataset config can differ from train).
4. **Analyze**: feature importance / interpretability hooks (method depends on model).

## Data Model (What You Provide)

OneEHR starts from a single, long-form event table and (optionally) a patient static table and/or label table.
These are plain CSVs; there is no dataset registry step.

### `dynamic.csv` (required)

Unified longitudinal event table (long format). Required columns:

- `patient_id` — patient identifier (stringable)
- `event_time` — parseable by `pandas.to_datetime`
- `code` — measurement / concept name
- `value` — numeric or categorical

Notes:

- Timestamps are **irregular**; binning happens during preprocessing.
- `value` typing/curation happens in preprocessing (numeric strategy + categorical strategy).

### `static.csv` (optional)

One row per patient. Minimum:

- `patient_id`

All other columns are treated as static covariates.

### `label.csv` (optional)

Long-form label events. Required columns:

- `patient_id`
- `label_time`
- `label_code`
- `label_value`

If you do not have a label table, you can generate labels via a Python `label_fn` (recommended for rapid iteration).

## Example Dataset (Simulated)

The repo includes a tiny simulated dataset under `examples/`:

- `examples/dynamic.csv` (required)
- `examples/static.csv` (optional)
- `examples/label.csv` (optional)

The default `examples/experiment.toml` points to these files.

## Tasks and Prediction Modes

OneEHR is currently **single-task**.

Supported task kinds:

- `binary`
- `regression`

Supported prediction modes:

- `patient` (N-1): one prediction per patient
- `time` (N-N): one prediction per time bin

## Config (TOML) Overview

The primary entrypoint is a single experiment config, e.g. `examples/experiment.toml`.

High-level sections:

- `[dataset]`: file paths (dynamic/static/label or dynamic + `labels.fn`)
- `[preprocess]`: binning + feature building strategies
- `[task]`: `kind` and `prediction_mode`
- `[labels]`: optional label generation function
- `[split]`: patient-level group splitting strategy
- `[[models]]`: one or more models to train
- `[hpo]` and `[hpo_models.<name>]`: optional config-driven grid search
- `[trainer]`: generic training and selection behavior
- `[output]`: where run artifacts are written

## Splits (Leakage Prevention)

All splits are **patient-level group splits** (patients never span train/val/test).

Config patterns (see `examples/experiment.toml`):

- K-fold CV: `[split] kind = "kfold"`
- Prospective/time boundary split: `[split] kind = "time"` (still grouped by patient)

## Preprocessing Design (Irregular → Fixed-Time)

Deep-learning and many tabular baselines want regular sampling. OneEHR converts irregular EHR events into fixed bins:

- `preprocess.bin_size`: e.g. `"1h"`, `"1d"`
- Per bin and per code:
  - numeric values: aggregated via `preprocess.numeric_strategy` (e.g. mean)
  - categorical values: represented via `preprocess.categorical_strategy` (e.g. one-hot or counts)

### Code selection (feature vocabulary)

`preprocess.code_selection` controls which codes enter the feature space:

- `frequency`: top-k by frequency (`top_k_codes`)
- `all`: include all codes
- `list`: explicit `code_list = ["A", "B"]`
- `importance`: load top-k from an importance file (e.g., SHAP export)

## Labels via `label_fn` (Recommended)

You can generate labels from events via a Python function configured in TOML:

```toml
[labels]
fn = "examples/label_fn.py:build_labels"
bin_from_time_col = true
```

Guidance:

- For `task.prediction_mode = "time"`, return time-aligned labels; OneEHR will align to bin times using `preprocess.bin_size`.
- Treat the label function as part of the experiment definition: version it and keep it next to your config.

## Models

Configure one or more models:

```toml
[[models]]
name = "gru"

[[models]]
name = "xgboost"
```

One model can also be configured (legacy style), but `[[models]]` is recommended for consistency.

## Hyperparameter Search (Config-Driven Grid Search)

There is no separate `oneehr hpo` command. Hyperparameter search is part of `oneehr train` when enabled in config.

Shared HPO config lives in `[hpo]`, per-model overrides live in `[hpo_models.<model_name>]`:

```toml
[hpo]
enabled = true
metric = "val_loss"
mode = "min"
scope = "single" # single | per_split | cv_mean
grid = [
  ["trainer.lr", [1e-3, 3e-4]],
  ["trainer.batch_size", [16, 32]],
]

[hpo_models.xgboost]
enabled = true
metric = "val_auroc"
mode = "max"
grid = [
  ["model.xgboost.max_depth", [4, 6, 8]],
  ["model.xgboost.n_estimators", [200, 500]],
]
```

Notes:

- Grid keys are dotted paths into the config (e.g. `trainer.lr`, `model.xgboost.max_depth`).
- Scope:
  - `single`: tune on one split (default `splits[0]` or `hpo.tune_split`)
  - `per_split`: tune separately per split
  - `cv_mean`: select by averaging an aggregate metric across splits

## CLI Reference

Run `uv run oneehr --help` for authoritative flags. Current commands:

- `oneehr preprocess --config <toml>`: preprocessing + artifact materialization
- `oneehr train --config <toml> [--force]`: training + optional grid search + evaluation summaries
- `oneehr test --config <toml> [--run-dir <run>] [--test-dataset <toml>]`: evaluate trained run on external test data
- `oneehr analyze --config <toml> [--run-dir <run>] --method <xgboost|shap|attention>`: feature importance analysis

## Outputs (Run Directory Contract)

By default, run artifacts are written under `[output]`:

```toml
[output]
root = "logs"
run_name = "example"
```

Key artifacts you can rely on:

- Preprocess outputs:
  - `binned.parquet`: binned dynamic events (long format)
  - `features/dynamic/feature_columns.json`: dynamic feature schema
  - `features/static/static_all.parquet`: static matrix (if enabled)
  - `views/patient_tabular.parquet` or `views/time_tabular.parquet`: modeling-ready tabular views
  - `run_manifest.json`: single source of truth (artifact paths + schema)
- Train outputs:
  - `summary.csv`: per-split metrics
  - `paper_table.csv`: aggregated “paper-style” table with 95% CI
  - `preds/<model>/...`: predictions (if `output.save_preds = true`)
  - `hpo/<model>/...`: grid search trials and selected configs (if enabled)

## Common Recipes

### 1) Train a single model (no HPO)

Set one model and disable HPO:

```toml
[[models]]
name = "xgboost"

[hpo]
enabled = false
```

### 2) Time-level (N-N) prediction

```toml
[task]
prediction_mode = "time"
```

Make sure `labels.fn` produces time-level labels (with `label_time`) that match your binning logic.

### 3) External test set evaluation

Use `oneehr test` with an external dataset config:

```bash
uv run oneehr test --config examples/experiment.toml --test-dataset path/to/test_dataset.toml
```

## Development Notes

- The CLI entrypoint is `oneehr` (see `pyproject.toml`).
- If you are contributing or using coding agents, read `AGENTS.md` for repository conventions and “do not break” design invariants.
