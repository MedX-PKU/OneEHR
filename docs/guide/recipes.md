# Recipes

Common workflow patterns for OneEHR experiments.

---

## Single model, no HPO

The simplest setup: one model with default hyperparameters.

```toml
[[models]]
name = "xgboost"

[hpo]
enabled = false
```

```bash
uv run oneehr preprocess --config experiment.toml
uv run oneehr train --config experiment.toml
```

---

## Multi-model benchmark

Compare multiple models on the same splits:

```toml
[[models]]
name = "gru"

[[models]]
name = "lstm"

[[models]]
name = "transformer"

[[models]]
name = "xgboost"

[[models]]
name = "catboost"
```

All models share the same patient-level splits, so metrics are directly comparable.

---

## Time-level (N-N) prediction

Predict at every time step instead of once per patient:

```toml
[task]
kind = "binary"
prediction_mode = "time"

[labels]
fn = "path/to/label_fn.py:build_labels"
bin_from_time_col = true
```

Ensure your `label_fn` returns time-aligned labels with `label_time`, `label`, and `mask` columns.

!!! note
    TCN only supports time mode. Most other DL models support both patient and time modes.

---

## Prospective evaluation with time split

Use a temporal boundary to simulate real-world deployment:

```toml
[split]
kind = "time"
time_boundary = "2020-01-01"
```

Patients before the boundary are used for training; patients after are the prospective test set.

### With nested CV on the training pool

```toml
[split]
kind = "time"
time_boundary = "2020-01-01"
inner_kind = "kfold"
inner_n_splits = 5
```

This runs 5-fold CV within the pre-boundary patients, then evaluates on the post-boundary test set.

---

## HPO with CV-mean selection

Select hyperparameters by averaging a metric across all CV folds:

```toml
[split]
kind = "kfold"
n_splits = 5

[hpo]
enabled = true
scope = "cv_mean"
aggregate_metric = "auroc"
grid = [
  ["trainer.lr", [1e-3, 3e-4, 1e-4]],
  ["trainer.batch_size", [32, 64]],
]
```

---

## External test set evaluation

Train on one dataset, evaluate on another:

```bash
# Train
uv run oneehr preprocess --config experiment.toml
uv run oneehr train --config experiment.toml

# Test on external data
uv run oneehr test --config experiment.toml --test-dataset test_dataset.toml
```

The test command re-bins the external data using the training-time code vocabulary, ensuring feature alignment.

---

## Feature importance workflow

Run analysis after training to understand model decisions:

```bash
# For XGBoost: native importance + SHAP
uv run oneehr analyze --config experiment.toml

# For DL models: specify method
uv run oneehr analyze --config experiment.toml --method shap
uv run oneehr analyze --config experiment.toml --method attention
```

---

## Calibrated predictions for deployment

Enable calibration for well-calibrated probabilities:

```toml
[calibration]
enabled = true
method = "temperature"
threshold_strategy = "f1"
use_calibrated = true
```

This fits a temperature scaling parameter on the validation set and selects an optimal F1 threshold.

---

## Using SHAP importance for code selection

Iterative workflow: train, analyze, then retrain with selected codes:

```bash
# Step 1: Train with all codes
uv run oneehr preprocess --config experiment.toml
uv run oneehr train --config experiment.toml
uv run oneehr analyze --config experiment.toml --method shap

# Step 2: Export SHAP importance, retrain with top codes
```

```toml
[preprocess]
code_selection = "importance"
top_k_codes = 50
importance_file = "analysis/shap_scores.csv"
```

---

## Static features with MCGRU/DrAgent

Use models with a dedicated static branch:

```toml
[dataset]
dynamic = "data/dynamic.csv"
static = "data/static.csv"

[[models]]
name = "mcgru"

[[models]]
name = "dragent"
```

These models process static and dynamic features through separate pathways. The `static_dim` is automatically derived from the run manifest.

---

## Quick dataset inspection

Preview your dataset without running a full experiment:

```bash
uv run oneehr preprocess --config experiment.toml --overview
```

This prints a JSON summary including patient count, event count, time range, and top codes by frequency.
