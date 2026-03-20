# Artifacts Reference

OneEHR writes all experiment outputs to a structured run directory under `{output.root}/{output.run_name}/`.

---

## Run Directory Layout

```text
{output.root}/{output.run_name}/
    manifest.json
    preprocess/
        binned.parquet
        labels.parquet
        split.json
        static.parquet          (when static.csv is provided)
    train/
        {model_name}/
            checkpoint.ckpt
            meta.json
    test/
        predictions.parquet
        metrics.json
    analyze/
        comparison.json
        feature_importance.json
```

---

## `manifest.json`

The single source of truth for the run, written by `oneehr preprocess` and read by all downstream commands.

Key fields:

| Field | Description |
|-------|-------------|
| `config` | Full experiment config snapshot |
| `feature_columns` | Dynamic feature column names from binning |
| `static_feature_columns` | Static feature column names (if static.csv provided) |
| `paths` | Relative paths to preprocess artifacts |

---

## Preprocess Artifacts

Written by `oneehr preprocess` under `preprocess/`.

### `binned.parquet`

Binned dynamic events in long format. Columns include `patient_id`, `bin_time`, and generated `num__*` / `cat__*` features.

### `labels.parquet`

Processed labels.

- Patient mode: `patient_id`, `label`
- Time mode: `patient_id`, `bin_time`, `label`, `mask`

### `split.json`

Patient-level split definition.

Fields:

- `train_patients`
- `val_patients`
- `test_patients`

### `static.parquet`

Encoded static feature matrix keyed by `patient_id`. Only present when `[dataset].static` is provided.

---

## Train Artifacts

Written by `oneehr train` under `train/{model_name}/`.

### `checkpoint.ckpt`

Serialized model checkpoint saved via `torch.save` (for all model types including tabular).

### `meta.json`

Model metadata used to rebuild the model.

Key fields:

| Field | Description |
|-------|-------------|
| `model_name` | Model config name (e.g. `"xgboost"`, `"gru"`) |
| `params` | Model hyperparameters from config |
| `train_metrics` | Metrics computed during training |
| `feature_columns` | Feature columns the model was trained on |

---

## Test Artifacts

Written by `oneehr test` under `test/`.

### `predictions.parquet`

Unified predictions from all trained models and configured systems.

Columns:

| Column | Description |
|--------|-------------|
| `system` | System/model name |
| `patient_id` | Patient identifier |
| `y_true` | Ground truth label |
| `y_pred` | Predicted value |

### `metrics.json`

Aggregated test metrics per system.

---

## Analyze Artifacts

Written by `oneehr analyze` under `analyze/`.

### `comparison.json`

Cross-system comparison metrics. Contains per-system metric breakdowns computed from `predictions.parquet`.

Key fields:

| Field | Description |
|-------|-------------|
| `module` | `"comparison"` |
| `task` | Task kind and prediction mode |
| `systems[]` | Per-system metrics (name, n, metrics dict) |

### `feature_importance.json`

Feature importance results per trained model.

Key fields:

| Field | Description |
|-------|-------------|
| `module` | `"feature_importance"` |
| `models` | Per-model importance (method, features, importances) |
