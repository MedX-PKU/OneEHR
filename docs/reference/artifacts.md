# Artifacts Reference

OneEHR writes all experiment outputs to a structured run directory under `{output.root}/{output.run_name}/`.

This page documents the current public artifact surface for the eval-first workflow. The core contract is the structured tree consumed by `train`, `test`, `analyze`, `eval`, `query`, and `webui`.

Typical writers:

- `preprocess` writes the run manifest, saved splits, binned data, labels, and tabular views
- `train` writes model checkpoints, split metrics, predictions, and HPO outputs
- `test` writes held-out or external-test outputs under `test_runs/`
- `analyze` writes `analysis/`
- `eval build`, `eval run`, and `eval report` write `eval/`

---

## Run directory layout

```text
{output.root}/{output.run_name}/
    run_manifest.json
    binned.parquet
    labels.parquet
    views/
        patient_tabular.parquet
        time_tabular.parquet
    features/
        static/
            static_all.parquet
    splits/
        {split_name}.json
    models/
        {model_name}/
            {split_name}/
                model.json | model.cbm | model.pkl | state_dict.ckpt
                model_meta.json
                metrics.json
    preds/
        {model_name}/
            {split_name}.parquet
    hpo/
        {model_name}/
            best_once.json
            trials_{split}.csv
    summary.json
    hpo_best.csv
    analysis/
        index.json
        {module}/
            summary.json
            *.csv
            plots/
                *.json
            cases/
                *.parquet
        comparison/
            summary.json
            train_metrics.csv
            test_metrics.csv
        feature_importance_{model}_{split}_{method}.json
    eval/
        index.json
        instances/
            instances.parquet
        evidence/
            {instance_dir}/
                evidence.json
                events.csv
                static.json
                analysis_refs.json
        predictions/
            {system_name}/
                predictions.parquet
        traces/
            {system_name}/
                trace.parquet
        summary.json
        reports/
            leaderboard.csv
            split_metrics.csv
            pairwise.csv
            summary.json
    test_runs/
    final/
```

---

## Core preprocess artifacts

Written by `oneehr preprocess` and read by all downstream commands.

### `run_manifest.json`

The single source of truth for the run.

Schema version: **5**

Key public fields:

| Field | Description |
|-------|-------------|
| `schema_version` | Manifest format version |
| `dataset` | Dataset paths used |
| `task` | Task config snapshot |
| `split` | Split config snapshot |
| `preprocess` | Preprocess config snapshot |
| `features.dynamic.feature_columns` | Dynamic feature column names |
| `features.static.feature_columns` | Static feature column names |
| `artifacts.binned_parquet_path` | Relative path to `binned.parquet` |
| `artifacts.labels_parquet_path` | Relative path to `labels.parquet` |
| `artifacts.patient_tabular_parquet_path` | Relative path to the patient view, when available |
| `artifacts.time_tabular_parquet_path` | Relative path to the time view, when available |

### `binned.parquet`

Binned dynamic events in long format. Columns include `patient_id`, `bin_time`, and generated `num__*` / `cat__*` features.

### `labels.parquet`

Processed labels.

- patient mode: `patient_id`, `label`
- time mode: `patient_id`, `bin_time`, `label`, `mask`

### `views/patient_tabular.parquet`

Model-ready patient-level view for N-1 prediction.

### `views/time_tabular.parquet`

Model-ready time-level view for N-N prediction.

### `features/static/static_all.parquet`

Encoded static feature matrix keyed by `patient_id`.

---

## Split artifacts

### `splits/{split_name}.json`

Patient-level split definition used by training, testing, and eval instance construction.

Fields:

- `name`
- `train_patients`
- `val_patients`
- `test_patients`

---

## Train artifacts

Written by `oneehr train`.

### `models/{model_name}/{split_name}/`

Per-model, per-split training outputs.

| File | Models | Description |
|------|--------|-------------|
| `model.json` | XGBoost | Serialized XGBoost booster |
| `model.cbm` | CatBoost | CatBoost native format |
| `model.pkl` | RF, DT, GBDT | Pickled sklearn model |
| `state_dict.ckpt` | DL models | PyTorch weights |
| `model_meta.json` | DL models | Constructor kwargs used to rebuild the model |
| `metrics.json` | All | Per-split evaluation metrics |

### `preds/{model_name}/{split_name}.parquet`

Saved model predictions when `[output].save_preds = true`.

Common columns:

- `patient_id`
- `y_true`
- `y_pred`
- `split`

Time-level predictions also include `bin_time`.

### `summary.json`

Run-level training summary with one record per model and split.

### `hpo/{model_name}/best_once.json`

Best hyperparameter selection result for the model when HPO is enabled.

### `hpo/{model_name}/trials_{split}.csv`

All HPO trial rows for one split.

### `hpo_best.csv`

Flat CSV of best HPO settings across models.

---

## Test artifacts

Written by `oneehr test` under `test_runs/`.

| File | Description |
|------|-------------|
| `test_runs/metrics_{model}_{split}.json` | Per-split test metrics |
| `test_runs/preds_{model}_{split}.parquet` | Per-split test predictions |
| `test_runs/test_summary.json` | Aggregated test summary |
| `test_runs/test_summary.csv` | Optional flat CSV export when evaluable rows exist |

---

## Analysis artifacts

Written by `oneehr analyze`.

### `analysis/index.json`

Run-level analysis index.

Key fields:

| Field | Description |
|-------|-------------|
| `schema_version` | Analysis schema version |
| `run_name` | Run name from `[output].run_name` |
| `task` | Task kind and prediction mode |
| `modules` | Per-module statuses and artifact paths |
| `comparison` | Optional compare-run outputs |

### `analysis/{module}/summary.json`

Module-level summary for one analysis module such as:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `test_audit`
- `temporal_analysis`
- `interpretability`

### `analysis/{module}/*.csv`

Tabular exports for the module. Examples:

- `analysis/dataset_profile/top_codes.csv`
- `analysis/cohort_analysis/split_roles.csv`
- `analysis/prediction_audit/slices.csv`
- `analysis/test_audit/metric_summary.csv`
- `analysis/temporal_analysis/segments.csv`

### `analysis/{module}/plots/*.json`

Serialized plot specs written when `[analysis].save_plot_specs = true`.

### `analysis/{module}/cases/*.parquet`

Case-level audit exports used for module drill-down. A common example is highest-error rows saved by `prediction_audit`.

### `analysis/comparison/summary.json`

Summary of compare-run outputs when `--compare-run` is provided.

### `analysis/comparison/train_metrics.csv`

Metric deltas between two runs' top-level `summary.json` files.

### `analysis/comparison/test_metrics.csv`

Metric deltas between two runs' `test_runs/test_summary.json` files when both exist.

### `analysis/feature_importance_{model}_{split}_{method}.json`

Compatibility export written by `interpretability`.

Fields:

| Field | Description |
|-------|-------------|
| `method` | Analysis method: `xgboost`, `shap`, or `attention` |
| `input_kind` | Input shape used: `2d` or `3d` |
| `feature_names` | Feature column names |
| `importances` | Importance scores aligned to `feature_names` |

---

## Eval artifacts

Written by `oneehr eval build`, `oneehr eval run`, and `oneehr eval report`.

### `eval/index.json`

Run-level eval index describing the frozen comparison set.

Key fields:

- `schema_version`
- `run_dir`
- `task`
- `eval.instance_unit`
- `eval.instance_path`
- `instance_count`
- `records[]`

Each `records[]` item can include:

- `instance_id`
- `patient_id`
- `split`
- `split_role`
- `prediction_mode`
- `bin_time`
- `ground_truth`
- `event_count`
- `static_feature_count`
- `evidence_path`

### `eval/instances/instances.parquet`

Frozen instance table used by every scored system.

Common columns:

- `instance_id`
- `split`
- `split_role`
- `patient_id`
- `prediction_mode`
- `task_kind`
- `ground_truth`
- `event_count`
- `first_event_time`
- `last_event_time`
- `has_static`

Time-level runs also include `bin_time`.

### `eval/evidence/{instance_dir}/evidence.json`

Instance-level metadata plus artifact pointers for the frozen evidence bundle.

### `eval/evidence/{instance_dir}/events.csv`

Leakage-safe event timeline aligned to the frozen instance.

### `eval/evidence/{instance_dir}/static.json`

Static features for the instance. When `[eval].include_static = false`, `features` may be empty.

### `eval/evidence/{instance_dir}/analysis_refs.json`

Optional analysis references attached when `[eval].include_analysis_context = true`.

### `eval/predictions/{system_name}/predictions.parquet`

Per-system prediction rows written by `eval run`.

Common columns:

- `instance_id`
- `patient_id`
- `split`
- `split_role`
- `bin_time`
- `ground_truth`
- `system_name`
- `system_kind`
- `framework_type`
- `parsed_ok`
- `prediction`
- `label`
- `probability`
- `value`
- `explanation`
- `confidence`
- `latency_ms`
- `token_usage_prompt`
- `token_usage_completion`
- `token_usage_total`
- `cost_usd`
- `round_count`
- `trace_row_count`
- `config_sha256`
- `framework_metadata_json`
- `error_code`
- `error_message`

### `eval/traces/{system_name}/trace.parquet`

Structured stage-level execution trace for framework systems.

Common columns:

- `instance_id`
- `patient_id`
- `split`
- `bin_time`
- `system_name`
- `framework_type`
- `backend_name`
- `provider_model`
- `round`
- `stage`
- `role`
- `actor_name`
- `parsed_ok`
- `prompt`
- `raw_response`
- `prompt_sha256`
- `response_sha256`
- `stage_output_json`
- `latency_ms`
- `token_usage_prompt`
- `token_usage_completion`
- `token_usage_total`
- `cost_usd`
- `error_code`
- `error_message`
- `config_sha256`

### `eval/summary.json`

Run-level summary for every enabled system.

Each record can include:

- `system_name`
- `system_kind`
- `framework_type`
- `row_count`
- `parsed_ok_rows`
- `coverage`
- `mean_latency_ms`
- `total_tokens`
- `total_cost_usd`
- `artifacts.predictions_parquet`
- `artifacts.trace_parquet`

### `eval/reports/leaderboard.csv`

One row per system with the primary ranking metric plus coverage and cost/latency fields.

### `eval/reports/split_metrics.csv`

Per-system, per-split metric table.

### `eval/reports/pairwise.csv`

Paired deltas and bootstrap summaries for configured system comparisons.

### `eval/reports/summary.json`

High-level report metadata such as:

- `primary_metric`
- `leaderboard_rows`
- `pairwise_rows`
- `artifacts.leaderboard_csv`
- `artifacts.split_metrics_csv`
- `artifacts.pairwise_csv`

---

## Final evaluation artifacts

Written for time-split prospective evaluation under `final/`.

| File | Description |
|------|-------------|
| `final/test_metrics_{model}.json` | Prospective test metrics |
| `final/test_preds_{model}.parquet` | Prospective test predictions |
| `final/test_summary.json` | Aggregated prospective summary when final evaluation runs |
