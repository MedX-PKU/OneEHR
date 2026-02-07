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

## How You Should Work In This Repo

## Repo Map (Where Things Live)

- `oneehr/cli/`: CLI entrypoints and command implementations
  - `oneehr/cli/main.py`: argparse wiring (subcommands)
  - `oneehr/cli/preprocess.py`: preprocessing pipeline
  - `oneehr/cli/train.py`: training + optional HPO grid search
  - `oneehr/cli/test.py`: external test-set evaluation
  - `oneehr/cli/analyze.py`: analysis hooks
- `oneehr/config/`: TOML schema + loader
  - `oneehr/config/schema.py`: dataclasses for config
  - `oneehr/config/load.py`: TOML parsing, defaults, validation
- `oneehr/hpo/`: minimal config-driven grid search utilities (used by `train`)
- `examples/`: runnable reference configs and templates
- `tests/`: unit/integration tests (keep them fast)

## Change Checklist (Do This Before You Finish)

- Update docs if CLI/config/artifacts change (`README.md` and/or `examples/*.toml`).
- Add/adjust tests for user-visible behavior.
- Run:
  - `uv run pytest -q`
  - `uv run oneehr --help`
  - `uv run oneehr preprocess --config examples/experiment.toml --overview` (quick smoke; avoids long training)

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

## Documentation Rules (Keep README Trustworthy)

- Do not document commands that do not exist. The authoritative source is `uv run oneehr --help`.
- If a workflow step changes (e.g. HPO behavior), update:
  - `README.md` (user workflow)
  - `examples/experiment.toml` (reference config)
  - any CLI `--help` text if needed

## Design Notes (Implementation Intent)

- **Preprocessing** turns irregular events into fixed-time bins (`preprocess.bin_size`) and materializes tabular “views”.
- **Training** may run a small grid search (HPO) *inside* `oneehr train` controlled by `[hpo]` and `[hpo_models.*]`.
- **Testing** is for external evaluation runs; it should read the run directory contract (manifest/artifacts) rather than re-deriving schema.
- **Analysis** methods are model-dependent; avoid claiming model-agnostic interpretability unless implemented.
