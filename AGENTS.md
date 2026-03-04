# OneEHR — Agent Working Agreement (AGENTS.md)

This file is optimized for Agent to onboard fast and operate safely.

## Project Snapshot

- Name: **OneEHR**
- Purpose: **EHR predictive modeling + analysis library in Python**
- CLI: `oneehr`
- Config: **TOML-first**
- Python: **3.12** (`.python-version`)
- Dependency manager: **uv** (required)

## What Agent Should Optimize For

**Primary goal:** help users run EHR experiments quickly and correctly, without introducing leakage or silent behavior changes.

When changing code, preserve these invariants:

- **Leakage prevention is non-negotiable**: all splits are patient-level group splits; no patient appears in multiple splits.
- **TOML is the source of truth**: CLI flags should not become a second config system (flags are for paths/overrides only).
- **Artifact contract matters**: run directories under `[output]` are part of the public interface (see README “Outputs”).
- **Event-table-first**: the main user input is a long-form event table with irregular timestamps (`event_time`, `code`, `value`).

## Setup (uv-only)

```bash
uv venv .venv --python 3.12
uv pip install -e .
```

Sanity checks:

```bash
uv run oneehr --help
uv run pytest -q
```

## Core Product Decisions (don’t “simplify away”)

### Data assumptions

- Start from a single doctor-friendly CSV/Excel event table.
- Event-level EHR is irregular timestamps + (`code`, `value`).
- Default time column: `event_time`.
- `value` may be numeric or categorical; typing/curation happens in preprocessing.
- Deep-learning discretization uses fixed-time binning with user `bin_size` (e.g. `1h`, `1d`).

### Tasks

- Single-task only (for now)
- Task types: `binary`, `regression` (no multiclass yet)
- Prediction modes:
  - N-1 (`patient`)
  - N-N (`time`)
- N-N labels can be generated via user-provided `label_fn`.

### Splits (leakage prevention is non-negotiable)

- All split strategies must be **patient-level group split**
- Supported/desired:
  - `k-fold`
  - `train/val/test`
  - time-based split (still grouped by patient)

## Supported Models

| Model | Config key | Type | Patient (N-1) | Time (N-N) | Static branch |
|-------|-----------|------|:---:|:---:|:---:|
| XGBoost | `xgboost` | Tabular | Yes | Yes | -- |
| CatBoost | `catboost` | Tabular | Yes | Yes | -- |
| Random Forest | `rf` | Tabular | Yes | Yes | -- |
| Decision Tree | `dt` | Tabular | Yes | Yes | -- |
| GBDT | `gbdt` | Tabular | Yes | Yes | -- |
| GRU | `gru` | DL | Yes | Yes | No |
| LSTM | `lstm` | DL | Yes | Yes | No |
| RNN | `rnn` | DL | Yes | Yes | No |
| Transformer | `transformer` | DL | Yes | Yes | No |
| TCN | `tcn` | DL | Yes | Yes | No |
| MLP | `mlp` | DL | Yes | Yes | No |
| AdaCare | `adacare` | DL | Yes | Yes | No |
| StageNet | `stagenet` | DL | Yes | Yes | No |
| RETAIN | `retain` | DL | Yes | Yes | No |
| ConCare | `concare` | DL | Yes | Yes | Yes |
| GRASP | `grasp` | DL | Yes | Yes | Yes |
| MCGRU | `mcgru` | DL | Yes | Yes | Yes |
| DrAgent | `dragent` | DL | Yes | Yes | Yes |

Source of truth: `oneehr/config/schema.py` (config dataclasses) and `oneehr/models/registry.py` (model specs).

## How You Should Work In This Repo

## Repo Map (Where Things Live)

- `oneehr/cli/`: CLI entrypoints and command implementations
  - `oneehr/cli/main.py`: argparse wiring (subcommands)
  - `oneehr/cli/preprocess.py`: preprocessing pipeline
  - `oneehr/cli/train.py`: training + optional HPO grid search
  - `oneehr/cli/test.py`: external test-set evaluation
  - `oneehr/cli/analyze.py`: analysis hooks
  - `oneehr/cli/_common.py`: shared CLI helpers
  - `oneehr/cli/_train_dl.py`: DL training helpers (patient/time level)
  - `oneehr/cli/_train_eval.py`: evaluation and metric computation
  - `oneehr/cli/_train_hpo.py`: HPO integration for training
- `oneehr/config/`: TOML schema + loader
  - `oneehr/config/schema.py`: dataclasses for config (all 18 model configs + `ExperimentConfig`)
  - `oneehr/config/load.py`: TOML parsing, defaults, validation
- `oneehr/models/`: model implementations (one file per model)
  - `oneehr/models/registry.py`: model name -> implementation mapping
  - `oneehr/models/constants.py`: `TABULAR_MODELS`, `DL_MODELS`, `STATIC_BRANCH_MODELS` sets
  - `oneehr/models/tabular.py`: tabular model train/predict wrappers
  - `oneehr/models/utils.py`: shared model utilities
- `oneehr/data/`: data loading, binning, feature engineering
  - `oneehr/data/binning.py`: irregular events -> fixed-time bins
  - `oneehr/data/splits.py`: patient-level group split strategies
  - `oneehr/data/labels.py`: label function loading and normalization
  - `oneehr/data/postprocess.py`: post-split pipeline (standardize, impute, etc.)
  - `oneehr/data/static_postprocess.py`: static feature encoding
  - `oneehr/data/sequence.py`: build/pad 3D tensors for DL models
  - `oneehr/data/tabular.py`: build patient/time tabular views
  - `oneehr/data/features.py`: feature column utilities
  - `oneehr/data/io.py`: CSV/parquet I/O
  - `oneehr/data/patient_index.py`: patient index construction
  - `oneehr/data/overview_light.py`: dataset overview generation
- `oneehr/eval/`: evaluation and metrics
  - `oneehr/eval/metrics.py`: metric computation (binary + regression)
  - `oneehr/eval/calibration.py`: temperature/Platt scaling
  - `oneehr/eval/bootstrap.py`: bootstrap test evaluation
  - `oneehr/eval/tables.py`: paper-style summary table generation
- `oneehr/artifacts/`: run directory contract and I/O
  - `oneehr/artifacts/run_manifest.py`: write `run_manifest.json` (schema v2)
  - `oneehr/artifacts/read.py`: `RunManifest` reader class
  - `oneehr/artifacts/materialize.py`: preprocess artifact materialization
  - `oneehr/artifacts/inference.py`: inference helpers for test pipeline
  - `oneehr/artifacts/labels.py`: label artifact I/O
  - `oneehr/artifacts/run_io.py`: run directory utilities
- `oneehr/analysis/`: interpretability hooks
  - `oneehr/analysis/feature_importance.py`: xgboost/shap/attention importance
- `oneehr/modeling/`: training infrastructure
  - `oneehr/modeling/trainer.py`: unified DL trainer (AdamW, early stopping, masked loss)
  - `oneehr/modeling/persistence.py`: model save/load
- `oneehr/hpo/`: minimal config-driven grid search utilities (used by `train`)
  - `oneehr/hpo/grid.py`: grid iteration + config override application
  - `oneehr/hpo/runner.py`: HPO trial runner
- `oneehr/utils/`: shared utilities (I/O, time parsing, imports)
- `docs/`: MkDocs documentation site (MkDocs 2 pre-release; see below)
- `examples/`: runnable reference configs and templates
- `tests/`: unit/integration tests (keep them fast)

## Documentation

The docs site uses the [MkDocs 2 pre-release](https://www.encode.io/mkdocs/) and lives in `docs/` with root config in `mkdocs.toml`.

```bash
# Install docs dependencies
uv pip install -e ".[docs]"

# Local preview
uv run mkdocs serve          # http://127.0.0.1:5000

# Build
uv run mkdocs build
```

When changing CLI flags, config parameters, model behavior, or artifact layout, update the corresponding pages under `docs/reference/` and `docs/guide/`.

## Change Checklist (Do This Before You Finish)

- Update docs if CLI/config/artifacts change (`README.md`, `examples/*.toml`, and `docs/`).
- Add/adjust tests for user-visible behavior.
- Run:
  - `uv run pytest -q`
  - `uv run oneehr --help`
  - `uv run oneehr preprocess --config examples/experiment.toml --overview` (quick smoke; avoids long training)
  - `uv run mkdocs build` (verify docs build succeeds)

### Conventional commits (do this often)

Use Conventional Commits whenever you commit:

- `feat: ...`
- `fix: ...`
- `refactor: ...`
- `docs: ...`
- `test: ...`
- `chore: ...`

### Plan-first when uncertain

If any requirement is ambiguous: Ask me to confirm before implementing.

Keep the questions concrete and choose defaults only after confirmation.

## Documentation Rules (Keep README + Docs Trustworthy)

- Do not document commands that do not exist. The authoritative source is `uv run oneehr --help`.
- If a workflow step changes (e.g. HPO behavior), update:
  - `README.md` (user workflow)
  - `examples/experiment.toml` (reference config)
  - `docs/reference/` and `docs/guide/` pages as appropriate
  - any CLI `--help` text if needed

## Design Notes (Implementation Intent)

- **Preprocessing** turns irregular events into fixed-time bins (`preprocess.bin_size`) and materializes tabular “views”.
- **Training** may run a small grid search (HPO) *inside* `oneehr train` controlled by `[hpo]` and `[hpo_models.*]`.
- **Testing** is for external evaluation runs; it should read the run directory contract (manifest/artifacts) rather than re-deriving schema.
- **Analysis** methods are model-dependent; avoid claiming model-agnostic interpretability unless implemented.
