# Artifacts Reference

OneEHR writes all experiment outputs to a structured run directory under `{output.root}/{output.run_name}/`. This page documents the full directory layout and key files.

---

## Run directory layout

```
{output.root}/{output.run_name}/
    run_manifest.json                       # (1)
    binned.parquet                          # (2)
    labels.parquet                          # (3)
    views/
        patient_tabular.parquet             # (4)
        time_tabular.parquet                # (5)
    features/
        dynamic/
            feature_columns.json            # (6)
        static/
            feature_columns.json            # (7)
            static_all.parquet              # (8)
    models/
        {model_name}/
            {split_name}/
                model.json                  # (9)
                model.cbm                   # (10)
                model.pkl                   # (11)
                state_dict.ckpt             # (12)
                model_meta.json             # (13)
                metrics.json                # (14)
    preds/
        {model_name}/
            {split_name}.parquet            # (15)
    hpo/
        {model_name}/
            best_once.json                  # (16)
            trials_{split}.csv              # (17)
    summary.json                            # (18)
    hpo_best.csv                            # (19)
    analysis/
        feature_importance_{model}_{split}_{method}.json  # (20)
    test_runs/                              # (21)
    final/                                  # (22)
```

---

## Preprocess artifacts

These are written by `oneehr preprocess` and read by all downstream commands.

### `run_manifest.json` { #run-manifest }

The **single source of truth** for the run. Contains schema version, dataset paths, task/split/preprocess config snapshots, feature column lists, and artifact paths.

Schema version: **2**

Key fields:

| Field | Description |
|-------|-------------|
| `schema_version` | Manifest format version |
| `dataset` | Dataset paths used |
| `task` | Task config snapshot |
| `split` | Split config snapshot |
| `preprocess` | Preprocess config snapshot |
| `features.dynamic.feature_columns` | List of dynamic feature column names |
| `features.static.feature_columns` | List of static feature column names |
| `features.static.matrix_parquet_path` | Path to static feature matrix |
| `artifacts.binned_parquet_path` | Path to binned events |
| `artifacts.labels_parquet_path` | Path to labels |
| `artifacts.patient_tabular_parquet_path` | Path to patient-level view |
| `artifacts.time_tabular_parquet_path` | Path to time-level view |

### `binned.parquet`

Binned dynamic events in long format. Columns include `patient_id`, `bin_time`, and all `num__*` / `cat__*` feature columns.

### `labels.parquet`

Processed labels. For patient mode: `patient_id`, `label`. For time mode: `patient_id`, `bin_time`, `label`, `mask`.

### `views/patient_tabular.parquet`

Modeling-ready tabular view for patient-level (N-1) prediction. One row per patient with all dynamic features and the label column.

### `views/time_tabular.parquet`

Modeling-ready tabular view for time-level (N-N) prediction. One row per (patient, time bin) with all dynamic features, label, and mask columns.

### `features/dynamic/feature_columns.json`

JSON array of dynamic feature column names (e.g. `["num__heart_rate", "cat__diagnosis__A01"]`).

### `features/static/feature_columns.json`

JSON array of static feature column names after encoding (e.g. `["num__age", "cat__sex__M", "cat__sex__F"]`).

### `features/static/static_all.parquet`

Encoded static feature matrix. One row per patient with `patient_id` and all `num__*` / `cat__*` static columns.

---

## Train artifacts

Written by `oneehr train`.

### Model files

Per model and per split:

| File | Models | Description |
|------|--------|-------------|
| `model.json` | XGBoost | Serialized XGBoost booster |
| `model.cbm` | CatBoost | CatBoost native format |
| `model.pkl` | RF, DT, GBDT | Pickle file for sklearn models |
| `state_dict.ckpt` | All DL models | PyTorch state dict checkpoint |
| `model_meta.json` | All DL models | Model constructor kwargs for rebuilding |

### `metrics.json`

Per-split evaluation metrics. Contents depend on task type:

- **Binary**: `loss`, `auroc`, `auprc`, `f1`, `accuracy`, `precision`, `recall`
- **Regression**: `loss`, `rmse`, `mae`, `r2`

Also includes `threshold` (for binary) and calibration metrics if calibration is enabled.

### `preds/{model_name}/{split_name}.parquet`

Saved predictions (when `output.save_preds = true`). Columns:

- `patient_id`
- `y_true` (ground truth)
- `y_pred` (predicted probability or value)
- `split` (`train`, `val`, or `test`)

### HPO artifacts

| File | Description |
|------|-------------|
| `hpo/{model}/best_once.json` | Best hyperparameters from grid search |
| `hpo/{model}/trials_{split}.csv` | All trial results for a given split |

### Summary files

| File | Description |
|------|-------------|
| `summary.json` | Per-model, per-split metrics in structured format |
| `hpo_best.csv` | Best HPO config per model (when HPO is enabled) |

---

## Analysis artifacts

Written by `oneehr analyze`.

### `analysis/feature_importance_{model}_{split}_{method}.json`

Feature importance results. Fields:

| Field | Description |
|-------|-------------|
| `method` | Analysis method (`xgboost`, `shap`, or `attention`) |
| `input_kind` | Input shape used (`2d` or `3d`) |
| `feature_names` | List of feature column names |
| `importances` | List of importance scores (same order as `feature_names`) |

---

## Test artifacts

Written by `oneehr test` under `test_runs/`.

| File | Description |
|------|-------------|
| `test_runs/{model}/{split}/metrics.json` | Per-split test metrics |
| `test_runs/{model}/{split}/preds.parquet` | Test predictions |
| `test_summary.json` | Aggregated test summary |

---

## Final evaluation artifacts

Written for time-split prospective evaluation under `final/`.

| File | Description |
|------|-------------|
| `final/{model}/metrics.json` | Prospective test metrics |
| `final/{model}/preds.parquet` | Prospective test predictions |
