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
        index.json                          # (20)
        index.md                            # (21)
        index.html                          # (22)
        {module}/
            summary.json                    # (23)
            summary.md                      # (24)
            summary.html                    # (25)
            *.csv                           # (26)
            plots/
                *.json                      # (27)
            cases/
                *.parquet                   # (28)
        comparison/
            summary.json                    # (29)
            train_metrics.csv               # (30)
            llm_metrics.csv                 # (31)
        feature_importance_{model}_{split}_{method}.json  # (32)
    test_runs/                              # (33)
    final/                                  # (34)
    llm/
        instances/
            patient_instances.parquet       # (35)
            time_instances.parquet          # (36)
            summary.json                    # (37)
        prompts/
            {llm_model}/
                {split}.jsonl               # (38)
        responses/
            {llm_model}/
                {split}.jsonl               # (39)
        parsed/
            {llm_model}/
                {split}.parquet             # (40)
        preds/
            {llm_model}/
                {split}.parquet             # (41)
        failures/
            {llm_model}/
                {split}.jsonl               # (42)
        metrics/
            {llm_model}/
                {split}.json                # (43)
        summary.json                        # (44)
    workspace/
        index.json                          # (45)
        cases/
            {case_id}/
                workspace.json              # (46)
                events.csv                  # (47)
                static.json                 # (48)
                predictions.csv             # (49)
                analysis_refs.json          # (50)
    review/
        prompts/
            {review_model}/
                {split}.jsonl               # (51)
        responses/
            {review_model}/
                {split}.jsonl               # (52)
        parsed/
            {review_model}/
                {split}.parquet             # (53)
        failures/
            {review_model}/
                {split}.jsonl               # (54)
        metrics/
            {review_model}/
                {split}__{source}__{target_model}.json  # (55)
        summary.json                        # (56)
```

---

## Preprocess artifacts

These are written by `oneehr preprocess` and read by all downstream commands.

### `run_manifest.json` { #run-manifest }

The **single source of truth** for the run. Contains schema version, dataset paths, task/split/preprocess config snapshots, feature column lists, LLM config snapshots, workspace/review config snapshots, and artifact paths.

Schema version: **3**

Key fields:

| Field | Description |
|-------|-------------|
| `schema_version` | Manifest format version |
| `dataset` | Dataset paths used |
| `task` | Task config snapshot |
| `split` | Split config snapshot |
| `preprocess` | Preprocess config snapshot |
| `llm` | LLM workflow config snapshot |
| `workspace` | Workspace materialization config snapshot |
| `review` | Reviewer workflow config snapshot |
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

## LLM artifacts

Written by `oneehr llm-preprocess` and `oneehr llm-predict`.

### `llm/instances/patient_instances.parquet`

LLM evaluation instances for patient-level prediction. Each row corresponds to one held-out patient from one split.

Columns include:

- `instance_id`
- `split`
- `split_role` (currently always `test`)
- `patient_id`
- `task_kind`
- `ground_truth` (nullable)
- `event_count`
- `first_event_time`
- `last_event_time`
- `has_static`

### `llm/instances/time_instances.parquet`

LLM evaluation instances for time-level prediction. Each row corresponds to one held-out `(patient_id, bin_time)` label instance.

Columns include:

- `instance_id`
- `split`
- `split_role` (currently always `test`)
- `patient_id`
- `bin_time`
- `task_kind`
- `ground_truth`

### `llm/prompts/{llm_model}/{split}.jsonl`

Rendered prompts for one LLM backend and one split. Written when `llm.save_prompts = true`.

### `llm/responses/{llm_model}/{split}.jsonl`

Raw response payloads for one LLM backend and one split. Written when `llm.save_responses = true`.

### `llm/parsed/{llm_model}/{split}.parquet`

Parsed JSON outputs with normalized prediction fields. Written when `llm.save_parsed = true`.

### `llm/preds/{llm_model}/{split}.parquet`

Prediction rows with audit fields. Common columns:

- `instance_id`
- `patient_id`
- `split`
- `split_role`
- `ground_truth`
- `parsed_ok`
- `prediction`
- `probability` (binary)
- `value` (regression)
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

### `llm/failures/{llm_model}/{split}.jsonl`

Rows that failed request or JSON parsing, including `error_code`, `error_message`, and the raw response when available.

### `llm/metrics/{llm_model}/{split}.json`

Per-split LLM metrics plus audit coverage fields:

- `total_rows`
- `parsed_ok_rows`
- `parse_success_rate`
- `ground_truth_rows`
- `scored_rows`
- `coverage`
- `metrics` (binary or regression metrics on successfully parsed rows)

### `llm/summary.json`

Run-level summary for all `[[llm_models]]` and splits.

---

## Workspace artifacts

Written by `oneehr workspace`.

### `workspace/index.json`

Run-level index for the evidence-grounded case workspace.

Key fields:

- `schema_version`
- `case_count`
- `records[]` with `case_id`, `patient_id`, `split`, `prediction_mode`, `workspace_path`, and evidence counts

### `workspace/cases/{case_id}/workspace.json`

Case-level metadata, evidence counts, and relative paths to the materialized evidence bundle.

### `workspace/cases/{case_id}/events.csv`

Filtered, leakage-safe event timeline for the case. Patient-level cases include the observed history; time-level cases are truncated at `bin_time`.

### `workspace/cases/{case_id}/static.json`

Patient-level static features included in the case bundle.

### `workspace/cases/{case_id}/predictions.csv`

All matching train and/or LLM predictions for the case. Typical columns include:

- `source`
- `model_name`
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

### `workspace/cases/{case_id}/analysis_refs.json`

References to available analysis modules plus patient-level audit matches when present.

---

## Review artifacts

Written by `oneehr llm-review`.

### `review/prompts/{review_model}/{split}.jsonl`

Rendered reviewer prompts for one reviewer backend and one split.

### `review/responses/{review_model}/{split}.jsonl`

Raw reviewer responses for one reviewer backend and one split.

### `review/parsed/{review_model}/{split}.parquet`

Parsed structured reviewer outputs. Columns include:

- `review_id`
- `case_id`
- `patient_id`
- `target_source`
- `target_model_name`
- `parsed_ok`
- `supported`
- `clinically_grounded`
- `leakage_suspected`
- `needs_human_review`
- `overall_score`
- `review_summary`
- `key_evidence_json`
- `missing_evidence_json`

### `review/failures/{review_model}/{split}.jsonl`

Rows that failed reviewer request or JSON parsing, including `error_code`, `error_message`, and the raw response when available.

### `review/metrics/{review_model}/{split}__{source}__{target_model}.json`

Grouped reviewer metrics per split and reviewed prediction target. Fields include:

- `total_rows`
- `parsed_ok_rows`
- `parse_success_rate`
- `metrics.supported_rate`
- `metrics.clinically_grounded_rate`
- `metrics.leakage_suspected_rate`
- `metrics.needs_human_review_rate`
- `metrics.mean_overall_score`

### `review/summary.json`

Run-level summary for all `[[review_models]]`, grouped by split, prediction source, and target model.

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

### `analysis/index.json`

Run-level index for the analysis bundle.

Key fields:

| Field | Description |
|-------|-------------|
| `schema_version` | Analysis schema version |
| `run_name` | Run name from `[output].run_name` |
| `task` | Task kind and prediction mode |
| `modules` | Per-module artifact paths and statuses |
| `comparison` | Optional compare-run outputs |

### `analysis/{module}/summary.json`

Module-level summary for one of:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `temporal_analysis`
- `interpretability`
- `llm_audit`

Each summary includes `schema_version`, `module`, and `status`, plus module-specific summary fields.

### `analysis/{module}/*.csv`

Tabular exports for the module when `csv` output is enabled. Examples:

- `analysis/dataset_profile/top_codes.csv`
- `analysis/cohort_analysis/split_roles.csv`
- `analysis/prediction_audit/slices.csv`
- `analysis/temporal_analysis/segments.csv`
- `analysis/llm_audit/slices.csv`

### `analysis/{module}/plots/*.json`

Serialized plot specifications written when `[analysis].save_plot_specs = true`.

### `analysis/{module}/cases/*.parquet`

Case-level audit exports. Current uses include:

- `prediction_audit`: highest-error prediction rows per model/split
- `llm_audit`: failure rows per LLM model/split when parse or request failures exist

### `analysis/comparison/summary.json`

Written only when `--compare-run` is provided. Summarizes metric deltas between the current run and the comparison run.

### `analysis/comparison/train_metrics.csv`

Per-model metric deltas derived from the two runs' `summary.json` files.

### `analysis/comparison/llm_metrics.csv`

Per-LLM-model metric deltas derived from the two runs' `llm/summary.json` files, when present in both runs.

### `analysis/feature_importance_{model}_{split}_{method}.json`

Legacy compatibility export written by the `interpretability` module. Fields:

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
