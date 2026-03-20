# OneEHR Refactoring Delta: Phase 1-9 → Pipeline Redesign

> **Purpose:** Documents the pipeline & artifact architecture redesign that followed
> the Phase 1-9 cleanup. The cleanup simplified file structure; this redesign
> rewrites pipeline semantics, config schema, artifact layout, and CLI surface
> from first principles.

---

## 1. Design Decisions

| Decision | Before | After |
|----------|--------|-------|
| CLI commands | 5 (`preprocess`, `train`, `test`, `analyze`, `eval`) | **4** (`preprocess`, `train`, `test`, `analyze`) |
| Artifact format | CSV + Parquet + JSON | **Parquet + JSON only** |
| Splitting | k-fold, random, time, nested CV | **One split only** (random or time) |
| HPO | Grid search with 3 scopes | **Removed** (use different configs) |
| Model config | Per-model typed dataclasses (XGBoostConfig, GRUConfig, ...) | **`params: dict`** on ModelConfig |
| Checkpoint format | model-specific (state_dict.ckpt, model.json, model.cbm) | **`torch.save` for all** → `checkpoint.ckpt` |
| Predictions | Per-model per-split parquet files | **Unified `predictions.parquet`** with `system` column |
| Calibration | Temperature + Platt scaling | **Removed entirely** |
| Trainer fields | 17 fields (monitor, monitor_mode, loss_fn, repeat, ...) | **11 fields** |

---

## 2. CLI Surface

### Before (5 commands)

```
oneehr preprocess  --config FILE [--overview] [--overview-top-k-codes N]
oneehr train       --config FILE [--force]
oneehr test        --config FILE [--run-dir DIR] [--test-dataset FILE] [--force] [--out-dir DIR]
oneehr analyze     --config FILE [--run-dir DIR] [--module NAME]... [--compare-run DIR] [--case-limit N] [--method M]
oneehr eval        build|run|report|trace|instance --config FILE [--run-dir DIR]
```

### After (4 commands)

```
oneehr preprocess  --config experiment.toml
oneehr train       --config experiment.toml [--force]
oneehr test        --config experiment.toml [--force]
oneehr analyze     --config experiment.toml [--module NAME]
```

| Command | Input | Output | What it does |
|---------|-------|--------|--------------|
| `preprocess` | raw CSV | `preprocess/` | Bin features, generate labels, split patients |
| `train` | `preprocess/` | `train/` | Train ML/DL models, save checkpoints |
| `test` | `train/` + config | `test/` | Run ALL systems (trained + LLM) on test set |
| `analyze` | `test/` | `analyze/` | SHAP, fairness, cross-system comparison |

### Removed: `oneehr eval` (entire subcommand tree)

The `eval build|run|report|trace|instance` workflow is replaced by the unified
`test` command. Trained models and LLM systems now produce identical rows in
`predictions.parquet`. The `trace` and `instance` inspection commands are removed.

---

## 3. Artifact Structure

### Before

```
{root}/{run_name}/
├── run_manifest.json
├── binned.parquet
├── labels.parquet
├── splits/{name}.json              # one per fold
├── views/
│   ├── patient_tabular.parquet
│   └── time_tabular.parquet
├── features/static/static_all.parquet
├── models/{model}/{split}/
│   ├── model.json / model.cbm      # ML
│   ├── state_dict.ckpt              # DL
│   ├── model_meta.json
│   ├── feature_columns.json
│   └── metrics.json
├── preds/{model}/{split}.parquet
├── preprocess/{split}/pipeline.json
├── summary.json
├── hpo_best.csv
├── hpo/{model}/{split}/*.json
├── test_runs/
│   ├── test_summary.json
│   ├── test_summary.csv
│   ├── metrics_{model}_{split}.json
│   └── preds_{model}_{split}.parquet
└── analysis/
    ├── index.json
    └── {module}/
        ├── summary.json
        ├── *.csv
        ├── plots/*.json
        └── cases/*.parquet
```

**~50+ files per run**

### After

```
{root}/{run_name}/
├── manifest.json                    # config snapshot + feature columns + paths
├── preprocess/
│   ├── binned.parquet               # (patient_id, bin_time, num__*, cat__*, label)
│   ├── labels.parquet               # (patient_id, [bin_time], label, [mask])
│   ├── split.json                   # {train: [...], val: [...], test: [...]}
│   └── static.parquet               # optional: (patient_id index, num__*, cat__*)
├── train/
│   └── {model_name}/
│       ├── checkpoint.ckpt          # torch.save() — ALL models (DL + ML)
│       └── meta.json                # hyperparams, train_metrics, feature_columns
├── test/
│   ├── predictions.parquet          # ALL systems × test patients
│   └── metrics.json                 # per-system metrics
└── analyze/
    └── {module}.json                # one file per analysis module
```

**~10-15 files per run**

### predictions.parquet schema

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `system` | str | yes | "xgboost", "gru", "gpt4o_single_llm" |
| `patient_id` | str | yes | |
| `bin_time` | datetime | time mode | |
| `y_true` | float | yes | |
| `y_pred` | float | yes | probability or regression value |
| `confidence` | float | no | self-reported confidence |
| `explanation` | str | no | LLM reasoning, null for ML/DL |
| `latency_ms` | float | no | |

### metrics.json schema

```json
{
  "task": {"kind": "binary", "prediction_mode": "patient"},
  "systems": [
    {"name": "xgboost", "kind": "trained_model", "n": 200,
     "metrics": {"auroc": 0.85, "auprc": 0.72}},
    {"name": "gpt4o_single_llm", "kind": "llm", "n": 200,
     "metrics": {"auroc": 0.78},
     "cost": {"usd": 1.23, "prompt_tokens": 50000}}
  ]
}
```

---

## 4. Config Schema

### Before: 23+ frozen dataclasses

```
DatasetConfig, DynamicTableConfig, StaticTableConfig, LabelTableConfig,
DatasetsConfig, PreprocessConfig, TaskConfig, LabelsConfig, SplitConfig,
TrainerConfig, HPOConfig, CalibrationConfig, ModelConfig,
XGBoostConfig, CatBoostConfig, GRUConfig, LSTMConfig, TCNConfig,
TransformerConfig, EvalBackendConfig, EvalSystemConfig, EvalSuiteConfig,
EvalConfig, AnalysisConfig, OutputConfig, ExperimentConfig
```

### After: 9 frozen dataclasses

```
DatasetConfig, PreprocessConfig, TaskConfig, SplitConfig, ModelConfig,
TrainerConfig, SystemConfig, OutputConfig, ExperimentConfig
```

### Config TOML comparison

**Removed sections:** `[labels]`, `[hpo]`, `[hpo_models.*]`, `[calibration]`,
`[eval]`, `[eval.backends]`, `[eval.systems]`, `[eval.suites]`, `[analysis]`,
`[datasets]`, `[model]` (singular)

**Removed fields:**
- `SplitConfig`: `n_splits`, `fold_index`, `inner_kind`, `inner_n_splits`
- `TrainerConfig`: `monitor`, `monitor_mode`, `loss_fn`, `bootstrap_test`, `bootstrap_n`, `repeat`, `grad_clip_norm` → `grad_clip`
- `PreprocessConfig`: `min_code_count`, `code_list`, `importance_file`, `importance_code_col`, `importance_value_col`, `pipeline`
- `OutputConfig`: `save_preds`
- `ExperimentConfig`: `model` (singular), `datasets`, `labels`, `hpo`, `hpo_by_model`, `calibration`, `eval`, `analysis`, `_dynamic_dim`

**New:**
- `ModelConfig.params: dict` replaces per-model typed sub-configs
- `SystemConfig` replaces `EvalSystemConfig` + `EvalBackendConfig` (flattened)
- `TrainerConfig.patience` replaces `early_stopping_patience`
- `TrainerConfig.grad_clip` replaces `grad_clip_norm`

### Example config

```toml
[dataset]
dynamic = "data/dynamic.csv"
static  = "data/static.csv"
label   = "data/label.csv"

[preprocess]
bin_size = "1d"
numeric_strategy = "mean"
categorical_strategy = "onehot"
code_selection = "frequency"
top_k_codes = 100

[task]
kind = "binary"
prediction_mode = "patient"

[split]
kind = "random"
seed = 42
val_size = 0.1
test_size = 0.2

[[models]]
name = "xgboost"
[models.params]
max_depth = 6
n_estimators = 500

[[models]]
name = "gru"
[models.params]
hidden_dim = 128
num_layers = 1

[trainer]
device = "auto"
seed = 42
max_epochs = 30
batch_size = 64
lr = 1e-3
weight_decay = 0.0
grad_clip = 1.0
num_workers = 0
precision = "fp32"
early_stopping = true
patience = 5

[[systems]]
name = "gpt4o_single_llm"
kind = "llm"
framework = "single_llm"
backend = "openai"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"

[output]
root = "runs"
run_name = "exp001"
```

---

## 5. Files Changed

### Rewritten (core pipeline)

| File | Lines before | Lines after | What changed |
|------|-------------|-------------|--------------|
| `config/schema.py` | 323 | 82 | 23 dataclasses → 9, `params: dict` model config |
| `config/load.py` | 533 | 72 | Generic `_build_dataclass()`, no validation sprawl |
| `data/splits.py` | 247 | 88 | No k-fold, one Split, `split.json` not `splits/{name}.json` |
| `data/io.py` | 63 | 53 | Simple `Path | None` args instead of wrapper configs |
| `data/labels.py` | 84 | 68 | Removed `cfg.labels.fn` coupling |
| `data/binning.py` | 218 | 195 | Removed `importance` selection, `min_code_count` |
| `artifacts/manifest.py` | 172 | 40 | Simple config snapshot, no eval/static metadata |
| `artifacts/materialize.py` | 178 | 95 | Output to `preprocess/`, no `views/` dir |
| `cli/main.py` | 105 | 56 | 4 commands, no eval subparser |
| `cli/preprocess.py` | 41 | 17 | No overview, no label_fn |
| `cli/train.py` | 396 | 218 | Simple model loop, no folds/HPO/calibration |
| `cli/test.py` | 510 | 240 | Unified predictions.parquet for all systems |
| `cli/analyze.py` | 54 | 96 | Self-contained comparison + feature_importance modules |
| `models/__init__.py` | 137 | 95 | `build_dl_model()` from `params: dict` |
| `models/tree.py` | 191 | 110 | `params: dict` instead of typed configs |
| `training/trainer.py` | 276 | 220 | Higher-level `fit_model()` with sequence building |
| `training/persistence.py` | 63 | 42 | `torch.save(model)` for all, `checkpoint.ckpt + meta.json` |

### Deleted (16 files)

| File | Lines | Why |
|------|-------|-----|
| `cli/_common.py` | 31 | `resolve_run_root` / `require_manifest` inlined |
| `cli/_train_dl.py` | 219 | DL training folded into `cli/train.py` |
| `cli/_train_eval.py` | 95 | Calibration + threshold removed |
| `cli/_train_hpo.py` | 357 | HPO removed entirely |
| `cli/eval.py` | 139 | Eval subcommand tree removed |
| `training/hpo.py` | 87 | HPO removed |
| `eval/calibration.py` | 149 | Calibration removed |
| `eval/workflow.py` | 1948 | Eval build/run/report replaced by unified `test` |
| `eval/query.py` | 173 | Trace/instance inspection removed |
| `eval/tables.py` | 111 | Metric summarization tables removed |
| `data/overview_light.py` | 103 | Dataset profiling overview removed |
| `data/test_samples.py` | 147 | Test sample builder removed |
| `artifacts/store.py` | 386 | RunIO class replaced by simple manifest reader |
| `AGENTS.md` | 49 | Outdated agent documentation |

**Net change: −6,330 lines deleted, +1,402 lines added**

### Updated (non-core)

| File | What changed |
|------|--------------|
| `analysis/__init__.py` | Cleared re-exports of deleted reporting functions |
| `analysis/reporting.py` | Not deleted but now dead code (imports broken modules) |
| `examples/tjh/experiment.toml` | Rewritten for new config schema (TJH dataset) |
| `tests/test_cli_smoke.py` | Rewritten for new pipeline |
| `tests/test_cli_end_to_end_simulated.py` | Rewritten for new pipeline |
| `tests/test_train_cli_contract.py` | Rewritten for new pipeline |

---

## 6. What's NOT Changed

These modules are untouched and work as before:

| Module | Status |
|--------|--------|
| `agent/` (client, runtime, contracts, schema, templates) | Intact |
| `eval/metrics.py` (binary + regression metrics) | Intact |
| `eval/bootstrap.py` (bootstrap CIs) | Intact |
| `analysis/feature_importance.py` (SHAP + native) | Intact |
| `data/binning.py` (core binning logic) | Minor simplification |
| `data/sequence.py` (sequence building for DL) | Intact |
| `data/tabular.py` (feature transforms) | Intact |
| `models/recurrent.py` (GRU/LSTM) | Intact |
| `models/transformer.py` | Intact |
| `models/tcn.py` | Intact |
| `utils/__init__.py` | Intact |

---

## 7. Migration Guide

### Config migration

```toml
# BEFORE                              # AFTER
[model]                                [[models]]
name = "xgboost"                       name = "xgboost"
[model.xgboost]                        [models.params]
max_depth = 6                          max_depth = 6

[split]                                [split]
kind = "kfold"                         kind = "random"
n_splits = 5                           # (no n_splits, no fold_index)

[hpo]                                  # (removed — use different configs)
enabled = true
grid = [...]

[calibration]                          # (removed entirely)
enabled = true

[labels]                               # (removed — labels from label CSV)
fn = "label_fn.py:build_labels"

[eval]                                 [[systems]]
[[eval.backends]]                      name = "gpt4o_single_llm"
[[eval.systems]]                       kind = "llm"
                                       backend = "openai"
                                       model = "gpt-4o"
```

### Artifact migration

| Before | After |
|--------|-------|
| `run_manifest.json` | `manifest.json` |
| `binned.parquet` (root) | `preprocess/binned.parquet` |
| `labels.parquet` (root) | `preprocess/labels.parquet` |
| `splits/{name}.json` | `preprocess/split.json` |
| `views/*.parquet` | (removed — tabular built on-the-fly) |
| `features/static/static_all.parquet` | `preprocess/static.parquet` |
| `models/{model}/{split}/` | `train/{model}/` |
| `preds/{model}/{split}.parquet` | `test/predictions.parquet` |
| `test_runs/` | `test/` |
| `summary.json`, `hpo_best.csv` | (removed) |
| `analysis/{module}/` (dir with multiple files) | `analyze/{module}.json` |

---

## 8. Current File Inventory

```
oneehr/                          (~42 files)
├── __init__.py
├── agent/                       (6 files — INTACT)
│   ├── __init__.py
│   ├── client.py
│   ├── contracts.py
│   ├── runtime.py
│   ├── schema.py
│   └── templates.py
├── analysis/                    (3 files)
│   ├── __init__.py
│   ├── feature_importance.py
│   └── reporting.py             (dead code — to be cleaned up)
├── artifacts/                   (3 files)
│   ├── __init__.py
│   ├── manifest.py              (rewritten: simple config snapshot)
│   └── materialize.py           (rewritten: outputs to preprocess/)
├── cli/                         (5 files)
│   ├── __init__.py
│   ├── analyze.py               (rewritten: reads predictions.parquet)
│   ├── main.py                  (rewritten: 4 commands)
│   ├── preprocess.py            (rewritten: simplified)
│   ├── test.py                  (rewritten: unified predictions)
│   └── train.py                 (rewritten: simple model loop)
├── config/                      (3 files)
│   ├── __init__.py
│   ├── load.py                  (rewritten: generic _build_dataclass)
│   └── schema.py                (rewritten: 9 dataclasses)
├── data/                        (6 files)
│   ├── __init__.py
│   ├── binning.py               (simplified)
│   ├── io.py                    (simplified: Path args)
│   ├── labels.py                (simplified: no label_fn coupling)
│   ├── sequence.py
│   ├── splits.py                (rewritten: one split only)
│   └── tabular.py
├── eval/                        (3 files)
│   ├── __init__.py
│   ├── bootstrap.py
│   └── metrics.py
├── models/                      (5 files)
│   ├── __init__.py              (rewritten: build_dl_model from params dict)
│   ├── recurrent.py
│   ├── tcn.py
│   ├── transformer.py
│   └── tree.py                  (rewritten: params dict)
├── training/                    (3 files)
│   ├── __init__.py
│   ├── persistence.py           (rewritten: torch.save for all)
│   └── trainer.py               (rewritten: higher-level fit_model)
└── utils/
    └── __init__.py
```
