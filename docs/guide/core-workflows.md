# Core Workflows

This guide covers the standard OneEHR operating path: prepare standardized tables, materialize features, train and test models, and write structured analysis.

Use this page for workflow decisions. Use the [Configuration Reference](../reference/configuration.md), [CLI Reference](../reference/cli.md), and [Artifacts Reference](../reference/artifacts.md) for full option tables and on-disk details.

Each command reads and writes the same shared run directory contract, so persisted state stays aligned across preprocessing, training, testing, and analysis.

## Workflow Shape

For a typical experiment, the command sequence is:

```bash
uv run oneehr preprocess --config experiment.toml
uv run oneehr train --config experiment.toml
uv run oneehr test --config experiment.toml
uv run oneehr analyze --config experiment.toml
```

All of these commands operate on the same run root, usually `{output.root}/{output.run_name}`.

## Preprocess

`oneehr preprocess` is the first required step for every run. It reads the standardized dataset tables, materializes the binned feature views, saves the split contract, and writes the run manifest used by downstream commands.

```bash
uv run oneehr preprocess --config experiment.toml
```

What preprocessing decides:

- Bin size and time alignment via `[preprocess].bin_size`
- Numeric and categorical aggregation strategies
- Code vocabulary selection via `[preprocess].code_selection` and `top_k_codes`
- Patient-level or time-level prediction mode from `[task].prediction_mode`
- Patient-level saved split under `preprocess/split.json`

Inputs come from `[dataset]`. The required raw shape is:

- `dynamic.csv`: `patient_id`, `event_time`, `code`, `value`
- `static.csv` optional: patient-level columns keyed by `patient_id`
- `label.csv` optional: `patient_id`, `label_time`, `label_code`, `label_value`

## Train

`oneehr train` fits one or more configured models against the saved preprocess artifacts and split contract.

```bash
uv run oneehr train --config experiment.toml
uv run oneehr train --config experiment.toml --force
```

Key behaviors:

- `[[models]]` selects the training targets
- Each model gets a `name` and optional `params` dict for hyperparameters
- Tabular and deep learning models use the same shared run contract
- Checkpoints and metadata are written under `train/{model_name}/`

In OneEHR, the TOML file is the experiment contract: if the config changes, the experiment changed.

## Test

`oneehr test` evaluates all trained models and configured `[[systems]]` on the held-out test split.

```bash
uv run oneehr test --config experiment.toml
uv run oneehr test --config experiment.toml --force
```

Outputs:

- `test/predictions.parquet` -- unified predictions with a `system` column for all models and systems
- `test/metrics.json` -- aggregated test metrics per system

Use `--force` to overwrite existing test outputs.

## Analyze

`oneehr analyze` reads `test/predictions.parquet` and writes structured analysis outputs under `analyze/`.

```bash
uv run oneehr analyze --config experiment.toml
uv run oneehr analyze --config experiment.toml --module comparison
```

Available modules:

- `comparison` -- cross-system metrics comparison
- `feature_importance` -- native importance for tree models, SHAP fallback for others

When `--module` is not specified, all available modules are run.

## Split Strategies

All supported split strategies are patient-level group splits. A patient never appears in more than one of train, validation, or test.

Supported strategies:

- `random` for a single random train/val/test partition
- `time` for a prospective patient-level split using `time_boundary`

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
