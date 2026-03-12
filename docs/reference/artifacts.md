# Artifacts Reference

OneEHR writes all experiment outputs to a structured run directory under `{output.root}/{output.run_name}/`.

The public contract is the structured artifact tree itself: JSON, JSONL, CSV, and Parquet. There is no markdown or HTML report layer in the current architecture.

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
        dynamic/
            feature_columns.json
        static/
            feature_columns.json
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
            agent_predict_metrics.csv
        feature_importance_{model}_{split}_{method}.json
    cases/
        index.json
        {case_dir}/
            case.json
            events.csv
            static.json
            predictions.csv
            analysis_refs.json
    agent/
        predict/
            instances/
                patient_instances.parquet | time_instances.parquet
                summary.json
            prompts/
                {predictor_name}/
                    {split}.jsonl
            responses/
                {predictor_name}/
                    {split}.jsonl
            parsed/
                {predictor_name}/
                    {split}.parquet
            preds/
                {predictor_name}/
                    {split}.parquet
            failures/
                {predictor_name}/
                    {split}.jsonl
            metrics/
                {predictor_name}/
                    {split}.json
            summary.json
        review/
            prompts/
                {reviewer_name}/
                    {split}.jsonl
            responses/
                {reviewer_name}/
                    {split}.jsonl
            parsed/
                {reviewer_name}/
                    {split}.parquet
            failures/
                {reviewer_name}/
                    {split}.jsonl
            metrics/
                {reviewer_name}/
                    {split}__{target_origin}__{target_predictor_name}.json
            summary.json
    test_runs/
    final/
```

---

## Core preprocess artifacts

Written by `oneehr preprocess` and read by all downstream commands.

### `run_manifest.json` { #run-manifest }

The single source of truth for the run.

Schema version: **4**

Key top-level fields:

| Field | Description |
|-------|-------------|
| `schema_version` | Manifest format version |
| `dataset` | Dataset paths used |
| `task` | Task config snapshot |
| `split` | Split config snapshot |
| `preprocess` | Preprocess config snapshot |
| `static` | Static postprocess pipeline snapshot |
| `cases` | Case materialization config snapshot |
| `agent` | Agent predict/review config snapshot tree |
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

- Patient mode: `patient_id`, `label`
- Time mode: `patient_id`, `bin_time`, `label`, `mask`

### `views/patient_tabular.parquet`

Model-ready patient-level view for N-1 prediction.

### `views/time_tabular.parquet`

Model-ready time-level view for N-N prediction.

### `features/dynamic/feature_columns.json`

JSON array of dynamic feature column names.

### `features/static/feature_columns.json`

JSON array of static feature column names after encoding.

### `features/static/static_all.parquet`

Encoded static feature matrix keyed by `patient_id`.

---

## Split artifacts

### `splits/{split_name}.json`

Patient-level split definition used by training, cases, and agent workflows.

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

## Cases artifacts

Written by `oneehr cases build`.

Case directories are filesystem-safe slugs with a short hash suffix. Treat `cases/index.json.records[].case_path` as the source of truth instead of constructing paths yourself.

### `cases/index.json`

Run-level case index.

Key fields:

- `schema_version`
- `run_dir`
- `case_count`
- `records[]` with `case_id`, `patient_id`, `split`, `prediction_mode`, `bin_time`, `ground_truth`, `case_path`, `event_count`, and `prediction_count`

### `cases/{case_dir}/case.json`

Case metadata plus artifact pointers.

Key fields include:

- `case_id`
- `patient_id`
- `split`
- `prediction_mode`
- `bin_time`
- `ground_truth`
- `artifacts`
- `evidence`

### `cases/{case_dir}/events.csv`

Leakage-safe event timeline for the case. Time-level cases are truncated at `bin_time`.

### `cases/{case_dir}/static.json`

Static features for the case. When `[cases].include_static = false`, `features` is empty.

### `cases/{case_dir}/predictions.csv`

Merged prediction table for the case across trained models and agent predictors.

Common columns:

- `origin`
- `predictor_name`
- `split`
- `patient_id`
- `prediction`
- `probability`
- `value`
- `confidence`
- `explanation`
- `parsed_ok`
- `error_code`
- `ground_truth`

Time-level cases also include `bin_time`.

Semantics:

- `origin = "model"` means a trained predictive model from `preds/`
- `origin = "agent"` means an agent backend result from `agent/predict/preds/`
- `predictor_name` is the concrete model or backend name, for example `xgboost` or `gpt4o-mini`

### `cases/{case_dir}/analysis_refs.json`

References to available analysis modules plus patient-level matches in `prediction_audit` and `agent_audit`, when present. When `[cases].include_analysis_refs = false`, the file still exists with empty lists.

---

## Agent prediction artifacts

Written by `oneehr agent predict`.

### `agent/predict/instances/patient_instances.parquet`

Agent prediction instances for patient-level runs.

Common columns:

- `instance_id`
- `split`
- `split_role`
- `patient_id`
- `task_kind`
- `ground_truth`
- `event_count`
- `first_event_time`
- `last_event_time`
- `has_static`

### `agent/predict/instances/time_instances.parquet`

Agent prediction instances for time-level runs. Includes `bin_time`.

### `agent/predict/instances/summary.json`

Summary of the materialized instance table with `sample_unit`, row count, and split list.

### `agent/predict/prompts/{predictor_name}/{split}.jsonl`

Rendered prompts for one backend and one split. Written when `agent.predict.save_prompts = true`.

### `agent/predict/responses/{predictor_name}/{split}.jsonl`

Raw response payloads for one backend and one split. Written when `agent.predict.save_responses = true`.

### `agent/predict/parsed/{predictor_name}/{split}.parquet`

Parsed prediction outputs. Written when `agent.predict.save_parsed = true`.

### `agent/predict/preds/{predictor_name}/{split}.parquet`

Prediction rows with audit fields.

Common columns:

- `instance_id`
- `patient_id`
- `split`
- `split_role`
- `predictor_name`
- `ground_truth`
- `parsed_ok`
- `prediction`
- `probability`
- `value`
- `explanation`
- `confidence`
- `prompt_sha256`
- `response_sha256`
- `token_usage_prompt`
- `token_usage_completion`
- `token_usage_total`
- `latency_ms`
- `error_code`
- `error_message`

Time-level outputs also include `bin_time`.

### `agent/predict/failures/{predictor_name}/{split}.jsonl`

Rows that failed request execution or structured parsing.

### `agent/predict/metrics/{predictor_name}/{split}.json`

Per-split metrics and coverage fields:

- `total_rows`
- `parsed_ok_rows`
- `parse_success_rate`
- `ground_truth_rows`
- `scored_rows`
- `coverage`
- `metrics`

### `agent/predict/summary.json`

Run-level summary for all prediction backends and splits.

Each record includes:

- `predictor_name`
- `provider_model`
- `split`
- `task_kind`
- `prediction_mode`
- `metrics`
- `artifacts`

---

## Agent review artifacts

Written by `oneehr agent review`.

### `agent/review/prompts/{reviewer_name}/{split}.jsonl`

Rendered reviewer prompts for one reviewer backend and one split.

### `agent/review/responses/{reviewer_name}/{split}.jsonl`

Raw reviewer responses for one reviewer backend and one split.

### `agent/review/parsed/{reviewer_name}/{split}.parquet`

Parsed structured review outputs.

Common columns:

- `review_id`
- `case_id`
- `patient_id`
- `split`
- `target_origin`
- `target_predictor_name`
- `parsed_ok`
- `supported`
- `clinically_grounded`
- `leakage_suspected`
- `needs_human_review`
- `overall_score`
- `review_summary`
- `key_evidence_json`
- `missing_evidence_json`
- `ground_truth`
- `error_code`
- `error_message`

### `agent/review/failures/{reviewer_name}/{split}.jsonl`

Rows that failed reviewer request execution or structured parsing.

### `agent/review/metrics/{reviewer_name}/{split}__{target_origin}__{target_predictor_name}.json`

Grouped review metrics for one reviewed target within one split.

Fields include:

- `total_rows`
- `parsed_ok_rows`
- `parse_success_rate`
- `metrics.supported_rate`
- `metrics.clinically_grounded_rate`
- `metrics.leakage_suspected_rate`
- `metrics.needs_human_review_rate`
- `metrics.mean_overall_score`

### `agent/review/summary.json`

Run-level summary for all reviewer backends, grouped by `reviewer_name`, split, `target_origin`, and `target_predictor_name`.

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

Module-level summary for one of:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `temporal_analysis`
- `interpretability`
- `agent_audit`

### `analysis/{module}/*.csv`

Tabular exports for the module. Examples:

- `analysis/dataset_profile/top_codes.csv`
- `analysis/cohort_analysis/split_roles.csv`
- `analysis/prediction_audit/slices.csv`
- `analysis/temporal_analysis/segments.csv`
- `analysis/agent_audit/slices.csv`

### `analysis/{module}/plots/*.json`

Serialized plot specs written when `[analysis].save_plot_specs = true`.

### `analysis/{module}/cases/*.parquet`

Case-level audit exports.

Current uses:

- `prediction_audit`: highest-error prediction rows per model and split
- `agent_audit`: failure or low-coverage rows per agent predictor and split

### `analysis/comparison/summary.json`

Summary of compare-run outputs when `--compare-run` is provided.

### `analysis/comparison/train_metrics.csv`

Metric deltas between two runs' top-level `summary.json` files.

### `analysis/comparison/agent_predict_metrics.csv`

Metric deltas between two runs' `agent/predict/summary.json` files.

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
