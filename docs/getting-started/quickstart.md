# Quickstart

This guide walks you through a complete experiment in 5 minutes using the bundled example dataset.

---

## 1. Install

```bash
uv venv .venv --python 3.12
uv pip install -e .
```

## 2. Preprocess

Bin events, build features, and materialize tabular views:

```bash
uv run oneehr preprocess --config examples/experiment.toml
```

This reads the example CSV files (`examples/dynamic.csv`, `examples/static.csv`, `examples/label.csv`), bins events into 1-day bins, selects the top 100 codes by frequency, and writes all artifacts to `logs/example/`.

Optionally, inspect the dataset:

```bash
uv run oneehr preprocess --config examples/experiment.toml --overview
```

## 3. Train

Train models with optional HPO grid search:

```bash
uv run oneehr train --config examples/experiment.toml
```

The example config trains two models (GRU and XGBoost) with 3-fold cross-validation. HPO is enabled: the GRU searches over learning rate and batch size, while XGBoost searches over max depth and number of estimators.

After training, check the results:

- `logs/example/summary.json` -- per-model, per-split metrics
- `logs/example/hpo_best.csv` -- best hyperparameters per model

## 4. Analyze

Run the default modular analysis suite:

```bash
uv run oneehr analyze --config examples/experiment.toml
```

Results are written to `logs/example/analysis/`, including:

- `analysis/index.json`
- module summaries such as `analysis/prediction_audit/summary.json`
- CSV tables such as `analysis/prediction_audit/slices.csv`
- plot specs under `analysis/<module>/plots/`
- static Markdown/HTML reports when enabled

## 5. (Optional) Test on external data

If you have a separate test dataset:

```bash
uv run oneehr test --config examples/experiment.toml --test-dataset path/to/test.toml
```

---

## What's in the example config?

The example `experiment.toml` demonstrates:

| Section | Setting |
|---------|---------|
| `[dataset]` | Three example CSVs (dynamic, static, label) |
| `[preprocess]` | 1-day bins, mean numeric, one-hot categorical, top 100 codes |
| `[task]` | Binary classification, patient-level (N-1) |
| `[labels]` | Label function at `examples/label_fn.py:build_labels` |
| `[split]` | 3-fold CV, seed 42, 20% validation |
| `[[models]]` | GRU + XGBoost |
| `[hpo]` | Grid search over trainer hyperparameters |
| `[trainer]` | 30 epochs, early stopping, final refit on train+val |

See [Configuration Reference](../reference/configuration.md) for every parameter.

## Next steps

- [Data Model](data-model.md) -- understand the three-table input format
- [Preprocessing Guide](../guide/preprocessing.md) -- customize binning and feature building
- [Training Guide](../guide/training.md) -- multi-model training and evaluation
- [Models Reference](../reference/models.md) -- all 18 models and their parameters
