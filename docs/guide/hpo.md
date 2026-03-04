# HPO (Grid Search)

OneEHR includes a config-driven grid search that runs inside `oneehr train`. There is no separate HPO command -- enable it in your TOML config.

---

## Enabling HPO

```toml
[hpo]
enabled = true
metric = "val_loss"
mode = "min"
grid = [
  ["trainer.lr", [1e-3, 3e-4]],
  ["trainer.batch_size", [16, 32]],
]
```

This searches over all combinations of learning rate and batch size, selecting the configuration that minimizes validation loss.

---

## Grid keys

Grid keys are **dotted paths** into the config. Supported roots:

| Root | Example keys |
|------|-------------|
| `trainer.*` | `trainer.lr`, `trainer.batch_size`, `trainer.weight_decay` |
| `model.*` | `model.xgboost.max_depth`, `model.gru.hidden_dim` |
| `preprocess.*` | `preprocess.top_k_codes` |
| `split.*` | `split.val_size` |

Each grid axis is a `[key, [values]]` pair. The search evaluates all Cartesian products.

---

## Scope

The `scope` parameter controls how the best config is selected:

### `single` (default)

Tune on a single split. By default, uses the first split (e.g. `fold0`). Use `tune_split` to specify which:

```toml
[hpo]
scope = "single"
tune_split = "fold0"
```

### `per_split`

Tune independently per split. Each split gets its own best hyperparameters:

```toml
[hpo]
scope = "per_split"
```

### `cv_mean`

Select the config that has the best average metric across all CV splits. Use `aggregate_metric` to specify which metric to average:

```toml
[hpo]
scope = "cv_mean"
aggregate_metric = "auroc"   # for binary: auroc, auprc; for regression: rmse, mae
```

---

## Per-model HPO

Different models often need different search spaces. Use `[hpo_models.<name>]` to override HPO settings per model:

```toml
# Default HPO for DL models
[hpo]
enabled = true
metric = "val_loss"
mode = "min"
grid = [
  ["trainer.lr", [1e-3, 3e-4]],
]

# XGBoost-specific HPO
[hpo_models.xgboost]
enabled = true
metric = "val_auroc"
mode = "max"
grid = [
  ["model.xgboost.max_depth", [4, 6, 8]],
  ["model.xgboost.n_estimators", [200, 500]],
]
```

Per-model HPO configs completely override the default `[hpo]` for that model.

---

## How it works internally

1. `iter_grid(hpo)` generates all Cartesian products of the grid as flat override dicts
2. `apply_overrides(cfg, overrides)` applies each override set to the frozen config using `dataclasses.replace()`
3. For each trial: train the model, evaluate on the target metric
4. Select the best trial per the `scope` strategy
5. Re-train with the best config for the final model

---

## HPO artifacts

| File | Description |
|------|-------------|
| `hpo/{model}/best_once.json` | Best hyperparameters |
| `hpo/{model}/trials_{split}.csv` | All trial results for a split |
| `hpo_best.csv` | Summary of best configs per model |

---

## Example: full HPO config

```toml
[[models]]
name = "gru"

[[models]]
name = "xgboost"

[hpo]
enabled = true
metric = "val_loss"
mode = "min"
scope = "cv_mean"
aggregate_metric = "auroc"
grid = [
  ["trainer.lr", [1e-3, 3e-4, 1e-4]],
  ["trainer.batch_size", [32, 64]],
]

[hpo_models.xgboost]
enabled = true
metric = "val_auroc"
mode = "max"
scope = "single"
grid = [
  ["model.xgboost.max_depth", [4, 6, 8]],
  ["model.xgboost.n_estimators", [200, 500]],
  ["model.xgboost.learning_rate", [0.01, 0.05, 0.1]],
]
```
