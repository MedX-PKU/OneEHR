# Core Workflows

This guide covers the standard OneEHR operating path: prepare standardized tables, materialize features, train and test models, write structured analysis, and produce the artifacts consumed by the unified evaluation layer.

Use this page for workflow decisions. Use the [Configuration Reference](../reference/configuration.md), [CLI Reference](../reference/cli.md), and [Artifacts Reference](../reference/artifacts.md) for full option tables and on-disk details.

Operationally, this is the task-oriented surface of OneEHR. Persisted state stays artifact-first: each command reads and writes the same run directory contract.

## Workflow Shape

For a typical experiment, the command sequence is:

```bash
uv run oneehr preprocess --config experiment.toml --overview
uv run oneehr train --config experiment.toml
uv run oneehr test --config experiment.toml
uv run oneehr analyze --config experiment.toml
uv run oneehr eval build --config experiment.toml
uv run oneehr eval run --config experiment.toml
uv run oneehr eval report --config experiment.toml
uv run oneehr query runs describe --config experiment.toml
```

All of these commands operate on the same run root, usually `{output.root}/{output.run_name}`.

## Preprocess

`oneehr preprocess` is the first required step for every run. It reads the standardized dataset tables, materializes the feature views, saves the split contract, and writes the run manifest used by downstream workflows.

```bash
uv run oneehr preprocess --config experiment.toml --overview
```

Use `--overview` when you want a machine-readable summary of the incoming dataset and selected code vocabulary before training.

What preprocessing decides:

- bin size and time alignment via `[preprocess].bin_size`
- numeric and categorical aggregation strategies
- code vocabulary selection and optional importance-based filtering
- post-split feature pipeline definitions such as standardization and imputation
- patient-level or time-level view materialization for `[task].prediction_mode`
- patient-level saved splits under `splits/`, including repeat-expanded names when `trainer.repeat > 1`

Inputs come from `[dataset]` or `[datasets].train`. The required raw shape is:

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

- `[model]` or `[[models]]` selects the training targets
- tabular and deep learning models share the same run contract
- split-level metrics, predictions, and model artifacts are written under the run directory
- final refit, early stopping, repeated runs, and grid search all live in the same training entrypoint

OneEHR is TOML-first: if the config changes, the experiment changed.

## Test

`oneehr test` evaluates the configured ML/DL models after `train` completes.

```bash
uv run oneehr test --config experiment.toml
uv run oneehr test --config experiment.toml --run-dir logs/my_run
uv run oneehr test --config experiment.toml --test-dataset path/to/test_config.toml
```

Default mode evaluates against the run's saved split contract. External mode uses `[datasets].test` or `--test-dataset` and aligns the external tables to the training-time schema.

Use `--out-dir` when you want the test output somewhere other than the default `test_runs/` location under the run directory.

## Analyze

`oneehr analyze` writes structured analysis modules under `analysis/`.

```bash
uv run oneehr analyze --config experiment.toml
uv run oneehr analyze --config experiment.toml --module prediction_audit --module cohort_analysis
uv run oneehr analyze --config experiment.toml --compare-run logs/baseline_run
```

Current modules are:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `test_audit`
- `temporal_analysis`
- `interpretability`

The public contract is structured only: JSON summaries, CSV tables, parquet case slices, and plot specs. There is no markdown or HTML report layer in the current architecture.

## Eval-Ready Artifacts And Query

After `analyze`, the run is ready for the unified evaluation layer. `oneehr eval build` freezes the instances and evidence that every compared system will see.

```bash
uv run oneehr eval build --config experiment.toml
uv run oneehr eval build --config experiment.toml --run-dir logs/my_run --force
```

Use `query` as the read-only interface over existing artifacts:

```bash
uv run oneehr query runs describe --run-dir logs/example
uv run oneehr query analysis modules --run-dir logs/example
uv run oneehr query eval index --run-dir logs/example
uv run oneehr query eval report --run-dir logs/example
```

`query` prints JSON to stdout, so it is the preferred entrypoint for automation, notebooks, and non-browser consumers.

## Why Eval Comes After Core Workflows

The evaluation layer is intentionally downstream of preprocess, train, test, and analyze:

- `preprocess` and `split` define the leakage-safe sampling contract.
- `train` produces the conventional ML/DL baselines you may want to compare against LLM or agent systems.
- `test` gives held-out model evaluation outside the unified eval leaderboard.
- `analyze` emits structured context that can optionally be attached to eval evidence bundles.
- `eval build/run/report` then compares systems on the same frozen instances with the same scoring code.

If you only need the eval comparison workflow in detail, continue with [Evaluation Workflows](eval-workflows.md).

## Split Strategies

All supported split strategies are patient-level group splits. A patient never appears in more than one of train, validation, or test.

Supported strategies:

- `kfold` for patient-level cross-validation
- `random` for a single random train/val/test partition
- `time` for a prospective patient-level split using `time_boundary`

Nested evaluation is available via `inner_kind` and `inner_n_splits` on `time` or `random` splits.

```toml
[split]
kind = "kfold"
n_splits = 5
val_size = 0.2
seed = 42
```

```toml
[split]
kind = "time"
time_boundary = "2012-01-01"
inner_kind = "kfold"
inner_n_splits = 5
```

## HPO And Repeated Training

Grid search is configured inside `oneehr train` rather than through a separate command.

```toml
[hpo]
enabled = true
metric = "val_loss"
mode = "min"
grid = [
  ["trainer.lr", [1e-3, 3e-4]],
  ["trainer.batch_size", [32, 64]],
]
```

Use `[hpo_models.<name>]` when different models need different search spaces. Use `trainer.repeat` when you want repeated runs per split with different effective seeds.

```toml
[trainer]
repeat = 3
```

## Calibration

Calibration is supported for binary classification tasks only and runs after split-level evaluation.

```toml
[calibration]
enabled = true
method = "temperature"  # temperature | platt
source = "val"
threshold_strategy = "f1"
use_calibrated = true
```

Use temperature scaling when you want the lowest-risk default. Use Platt scaling when you need a more flexible logistic recalibration model and can tolerate a bit more overfitting risk on small validation sets.

## Label Functions

Use `[labels].fn` when labels should be computed from raw events instead of loaded from a precomputed label table.

```toml
[labels]
fn = "examples/label_fn.py:build_labels"
bin_from_time_col = true
```

The callable contract is:

```python
def build_labels(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg,
) -> pd.DataFrame:
    ...
```

Return requirements:

- patient mode: `patient_id`, `label`
- time mode: `patient_id`, `label_time`, `label`, `mask`

Use label functions when label logic should remain versioned with the experiment or depends on the raw event history.
