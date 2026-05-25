# Core Workflows

This guide explains the standard OneEHR path: prepare CSV tables, materialize features, train models, test systems, write analysis outputs, and render figures.

Use [Configuration Reference](../reference/configuration.md), [CLI Reference](../reference/cli.md), and [Artifacts Reference](../reference/artifacts.md) for complete option tables and file layouts.

## Command Sequence

```bash
oneehr preprocess --config experiment.toml
oneehr train      --config experiment.toml
oneehr test       --config experiment.toml
oneehr analyze    --config experiment.toml
oneehr plot       --config experiment.toml
```

All commands read or write the same run directory: `{output.root}/{output.run_name}`.

## Preprocess

`oneehr preprocess` reads standardized dataset tables and writes the artifacts used by all later stages.

```bash
uv run oneehr preprocess --config experiment.toml
```

Preprocessing uses:

- `[dataset]` paths for `dynamic.csv`, optional `static.csv`, and optional `label.csv`
- `[preprocess].bin_size` for time-window size
- `numeric_strategy` and `categorical_strategy` for event aggregation and encoding
- `code_selection`, `top_k_codes`, and `min_code_count` for feature vocabulary selection
- `[task].prediction_mode` for patient-level or time-level label alignment
- `[split]` for patient-level train/validation/test assignment
- `[preprocess].pipeline` for train-split-fitted preprocessing steps

Key outputs:

- `preprocess/binned.parquet`
- `preprocess/labels.parquet`
- `preprocess/split.json`
- `preprocess/static.parquet` when `static.csv` is provided
- `preprocess/feature_schema.json`
- `preprocess/obs_mask.parquet`
- `preprocess/fitted_pipeline.pt`

## Train

`oneehr train` fits every model listed in `[[models]]`.

```bash
uv run oneehr train --config experiment.toml
uv run oneehr train --config experiment.toml --force
```

Training reads the saved preprocess artifacts and writes one directory per model under `train/`.

Model config pattern:

```toml
[[models]]
name = "xgboost"
[models.params]
n_estimators = 100
max_depth = 4

[[models]]
name = "gru"
[models.params]
hidden_dim = 64
num_layers = 1
```

Deep learning training options live under `[trainer]`, including device, seed, epoch count, batch size, optimizer settings, precision, scheduler, class weighting, and early stopping.

## Test

`oneehr test` evaluates trained models and configured `[[systems]]` on the held-out test split.

```bash
uv run oneehr test --config experiment.toml
uv run oneehr test --config experiment.toml --force
```

Outputs:

- `test/predictions.parquet` - one prediction table with a `system` column
- `test/metrics.json` - aggregated metrics per model or system

The shared `predictions.parquet` format is what enables model and system outputs to be analyzed together.

## Analyze

`oneehr analyze` reads `test/predictions.parquet` and writes JSON outputs under `analyze/`.

```bash
uv run oneehr analyze --config experiment.toml
uv run oneehr analyze --config experiment.toml --module comparison
```

Available modules:

- `comparison` - metrics by system with bootstrap confidence intervals where supported
- `feature_importance` - native importance, SHAP, permutation importance, or integrated gradients depending on model type
- `fairness` - demographic parity, equalized odds, predictive parity, and SMD summaries for binary tasks
- `calibration` - calibration metrics and calibrated predictions for binary tasks
- `statistical_tests` - pairwise statistical tests and multiple-testing correction
- `missing_data` - missingness summaries from preprocessed features

When `--module` is omitted, all modules run.

## Plot

`oneehr plot` renders figures whose required artifacts exist in the run directory.

```bash
uv run oneehr plot --config experiment.toml --style nature
uv run oneehr plot --config experiment.toml --figure roc pr calibration
```

Registered figure names:

- `roc`
- `pr`
- `forest`
- `calibration`
- `feature_importance`
- `confusion`
- `training_curves`
- `fairness`
- `missing_data`
- `decision_curve`
- `significance`
- `cohort_flow`

Style presets: `default`, `nature`, `lancet`, `wide`.

## Split Strategies

All supported split strategies are patient-level group splits. A patient appears in only one of train, validation, or test.

Random split:

```toml
[split]
kind = "random"
seed = 42
val_size = 0.1
test_size = 0.2
```

Time split:

```toml
[split]
kind = "time"
time_boundary = "2012-01-01"
```
