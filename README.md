# OneEHR

OneEHR is an end-to-end EHR predictive modeling + analysis toolkit in Python, designed around:

- **Doctor-friendly input**: start from a single long-form event table (CSV/Excel).
- **Leakage prevention by default**: all data splits are patient-level group splits.
- **TOML-first experiments**: most behavior is configured via one `experiment.toml`.
- **CLI-first workflow**: preprocess → train/test/analyze for ML/DL, plus LLM-specific prompt materialization and prediction commands.

This README is the user-facing guide. For full documentation, see the [docs site](https://medx-pku.github.io/OneEHR/) or build locally with `uv run mkdocs serve` (MkDocs 2 pre-release + `mkdocs.toml`). If you are using agents to modify the repo, read `AGENTS.md` too.

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
The default analysis run writes a modular report bundle under `logs/example/analysis/`, including summaries, CSV tables, plot specs, and static Markdown/HTML reports.

For the LLM workflow, run:

```bash
uv run oneehr preprocess --config examples/experiment.toml
uv run oneehr llm-preprocess --config examples/experiment.toml
uv run oneehr llm-predict --config examples/experiment.toml
```

## The Workflow (Conceptual)

OneEHR is organized around an explicit pipeline:

1. **Preprocess**: validate and bin irregular events into fixed time bins; build features; materialize tabular “views”.
2. **Train**: fit one or more models; optionally run config-driven grid search (HPO) per model.
3. **Test**: evaluate a trained run on the held-out test split, or on external test data (if a different dataset config is provided).
4. **Analyze**: modular audit/reporting over the run directory, including dataset profiling, cohort drift, prediction audit, temporal slices, interpretability, and LLM audit.
5. **LLM Preprocess / Predict**: materialize patient-level or time-window prompt instances, render structured summaries, call OpenAI-compatible chat completions, and score parsed predictions.

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
- `[analysis]`: default analysis modules, report formats, and audit limits
- `[llm]`, `[llm.prompt]`, `[llm.output]`: LLM inference/evaluation behavior
- `[[llm_models]]`: one or more OpenAI-compatible chat completion endpoints/models
- `[output]`: where run artifacts are written

### LLM workflow

OneEHR now includes an LLM-specific inference path for EHR prediction:

- Input: raw EHR history rendered into the built-in `summary_v1` structured prompt
- Backend: OpenAI-compatible `chat/completions`
- Tasks: `binary` and `regression`
- Units: `patient` and `time`
- Output: strict JSON, parsed into prediction artifacts and evaluation summaries

The current LLM path is inference/evaluation only. It does not fine-tune models and does not implement tool-calling or multi-turn agents yet.

Minimal config shape:

```toml
[llm]
enabled = true
sample_unit = "patient"      # patient | time
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

[llm.prompt]
include_static = true
include_labels_context = false
max_events = 200
time_order = "asc"

[llm.output]
include_explanation = true
include_confidence = false

[[llm_models]]
name = "gpt4o-mini"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
supports_json_schema = true
```

Notes:

- `llm.sample_unit` must match `task.prediction_mode`.
- `labels` are optional for patient-level inference, but time-level LLM evaluation requires labels.
- LLM-only configs are allowed. You do not need `[model]` or `[[models]]` when the run is only for `preprocess`, `llm-preprocess`, and `llm-predict`.

### Analysis workflow

`oneehr analyze` now writes a modular analysis bundle instead of a single feature-importance file. The default module set is:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `temporal_analysis`
- `interpretability`
- `llm_audit`

Minimal config shape:

```toml
[analysis]
default_modules = [
  "dataset_profile",
  "cohort_analysis",
  "prediction_audit",
  "temporal_analysis",
  "interpretability",
  "llm_audit",
]
formats = ["json", "csv", "md", "html"]
top_k = 20
stratify_by = []
case_limit = 50
save_plot_specs = true
shap_max_samples = 500
```

Notes:

- `summary.json` and `analysis/index.json` are always written.
- `formats` controls which extra human-readable report outputs are emitted; the default writes CSV tables plus Markdown/HTML reports.
- `--method` remains a compatibility shortcut for the `interpretability` module.

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

OneEHR ships 18 models: 5 tabular and 13 deep learning.

| Model | Config key | Type | Patient (N-1) | Time (N-N) | Static branch |
|-------|-----------|------|:---:|:---:|:---:|
| XGBoost | `xgboost` | Tabular | Yes | Yes | -- |
| CatBoost | `catboost` | Tabular | Yes | Yes | -- |
| Random Forest | `rf` | Tabular | Yes | Yes | -- |
| Decision Tree | `dt` | Tabular | Yes | Yes | -- |
| GBDT | `gbdt` | Tabular | Yes | Yes | -- |
| GRU | `gru` | DL | Yes | Yes | No |
| LSTM | `lstm` | DL | Yes | Yes | No |
| RNN | `rnn` | DL | Yes | Yes | No |
| Transformer | `transformer` | DL | Yes | Yes | No |
| TCN | `tcn` | DL | Yes | Yes | No |
| MLP | `mlp` | DL | Yes | Yes | No |
| AdaCare | `adacare` | DL | Yes | Yes | No |
| StageNet | `stagenet` | DL | Yes | Yes | No |
| RETAIN | `retain` | DL | Yes | Yes | No |
| ConCare | `concare` | DL | Yes | Yes | Yes |
| GRASP | `grasp` | DL | Yes | Yes | Yes |
| MCGRU | `mcgru` | DL | Yes | Yes | Yes |
| DrAgent | `dragent` | DL | Yes | Yes | Yes |

Configure one or more models:

```toml
[[models]]
name = "gru"

[[models]]
name = "xgboost"
```

One model can also be configured (legacy style), but `[[models]]` is recommended for consistency. See the [Models Reference](https://medx-pku.github.io/OneEHR/reference/models/) for all per-model parameters.

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
- `oneehr test --config <toml> [--run-dir <run>] [--test-dataset <toml>]`: evaluate trained run on test data
- `oneehr analyze --config <toml> [--run-dir <run>] [--module <name>] [--format <fmt>] [--compare-run <run>] [--case-limit <n>] [--method <xgboost|shap|attention>]`: modular run analysis and static report generation
- `oneehr llm-preprocess --config <toml> [--run-dir <run>] [--force]`: materialize LLM prompt instances from grouped patient splits
- `oneehr llm-predict --config <toml> [--run-dir <run>] [--force]`: render prompts, call OpenAI-compatible chat completions, parse strict JSON, and write LLM evaluation artifacts

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
  - `summary.json`: structured per-model, per-split metrics
  - `hpo_best.csv`: best HPO config per model
  - `preds/<model>/...`: predictions (if `output.save_preds = true`)
  - `hpo/<model>/...`: grid search trials and selected configs (if enabled)
- Analysis outputs:
  - `analysis/index.json`: run-level analysis index
  - `analysis/<module>/summary.json`: module summary
  - `analysis/<module>/*.csv`: module tables (when `csv` format is enabled)
  - `analysis/<module>/plots/*.json`: serialized plot specs
  - `analysis/<module>/cases/*.parquet`: case-level audit exports
  - `analysis/comparison/*`: optional compare-run outputs
  - `analysis/feature_importance_{model}_{split}_{method}.json`: legacy compatibility exports from the interpretability module
- LLM outputs:
  - `llm/instances/*.parquet`: patient-level or time-level prompt instances
  - `llm/prompts/<llm_model>/*.jsonl`: rendered prompts (default on)
  - `llm/responses/<llm_model>/*.jsonl`: raw model responses (default on)
  - `llm/preds/<llm_model>/*.parquet`: parsed prediction rows
  - `llm/metrics/<llm_model>/*.json`: per-split metrics + parse coverage
  - `llm/summary.json`: run-level LLM summary

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

### 3) Test Set Evaluation

By default, `oneehr test` relies on the internal splits generated during training to evaluate the model on held-out test patients without data leakage:

```bash
uv run oneehr test --config examples/experiment.toml
```

If you have a completely separate external dataset, provide its config:

```bash
uv run oneehr test --config examples/experiment.toml --test-dataset path/to/external_test.toml
```

### 4) LLM EHR prediction

Use the grouped split artifacts from preprocessing, render `summary_v1` prompts, and evaluate one or more OpenAI-compatible models:

```bash
uv run oneehr preprocess --config examples/experiment.toml
uv run oneehr llm-preprocess --config examples/experiment.toml
uv run oneehr llm-predict --config examples/experiment.toml
```

This writes all LLM outputs under `logs/<run_name>/llm/`.

### 5) Modular analysis and static reports

Run the full default analysis suite:

```bash
uv run oneehr analyze --config examples/experiment.toml
```

Run only selected modules and compare against another run:

```bash
uv run oneehr analyze \
  --config examples/experiment.toml \
  --module prediction_audit \
  --module interpretability \
  --compare-run logs/baseline_run
```

The analysis command writes `analysis/index.json` plus per-module summaries, CSV tables, plot specs, and optional Markdown/HTML reports.

## Documentation

Full documentation is available at [medx-pku.github.io/OneEHR](https://medx-pku.github.io/OneEHR/) or can be built locally:

```bash
uv pip install -e ".[docs]"
uv run mkdocs serve          # http://127.0.0.1:5000
```

Docs deployment is automated by `.github/workflows/docs-deploy.yml` on pushes to `main`.

The docs cover:

- **Getting Started**: installation, quickstart, data model
- **User Guide**: preprocessing, training, testing, analysis, splits, HPO, calibration, label functions, recipes
- **Reference**: CLI flags, all TOML parameters, all 18 models, run directory artifacts

## Development Notes

- The CLI entrypoint is `oneehr` (see `pyproject.toml`).
- If you are contributing or using coding agents, read `AGENTS.md` for repository conventions and “do not break” design invariants.
