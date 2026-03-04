# Analysis

`oneehr analyze` computes feature importance for trained models. Three methods are available, each suited to different model types.

---

## Basic usage

```bash
uv run oneehr analyze --config experiment.toml
```

### Specify a method

```bash
uv run oneehr analyze --config experiment.toml --method shap
```

### Specify a run directory

```bash
uv run oneehr analyze --config experiment.toml --run-dir logs/my_run
```

---

## Methods

### `xgboost` -- native feature importance

Uses XGBoost's built-in `get_score()` with gain-based importance. Only works with XGBoost models.

```bash
uv run oneehr analyze --config experiment.toml --method xgboost
```

### `shap` -- SHAP values

Uses `shap.Explainer` to compute SHAP values. Works with any model. For sequence models, the last time step is used as a 2D tabular proxy.

```bash
uv run oneehr analyze --config experiment.toml --method shap
```

!!! note
    For binary classification with 3D SHAP output `(N, D, C)`, OneEHR takes the positive-class slice and averages over samples.

### `attention` -- attention-weighted features

For DL models that expose attention weights. Computes attention-weighted absolute feature values, averaged over batch and time.

```bash
uv run oneehr analyze --config experiment.toml --method attention
```

Requires attention weights of shape `(B, T)` and features of shape `(B, T, D)`.

---

## Method selection

If `--method` is not specified:

- For tabular models (XGBoost): runs both `xgboost` (native) and `shap` importance
- For other models: you must specify `--method`

---

## Output format

Results are written to `analysis/feature_importance_{model}_{split}_{method}.json`:

```json
{
  "method": "shap",
  "input_kind": "2d",
  "feature_names": ["num__heart_rate", "num__lab_glucose", ...],
  "importances": [0.342, 0.198, ...]
}
```

| Field | Description |
|-------|-------------|
| `method` | Analysis method used |
| `input_kind` | Input shape (`2d` for tabular/flattened, `3d` for sequence) |
| `feature_names` | Feature column names in order |
| `importances` | Importance scores (same order as `feature_names`) |
