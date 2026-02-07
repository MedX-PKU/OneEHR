# OneEHR — Codex Working Agreement (AGENTS.md)

This file is optimized for Codex to onboard fast and operate safely.

## Project Snapshot

- Name: **OneEHR**
- Purpose: **EHR predictive modeling + analysis library in Python**
- CLI: `oneehr`
- Config: **TOML-first**
- Python: **3.12** (`.python-version`)
- Dependency manager: **uv** (required)

## Setup (uv-only)

```bash
uv venv .venv --python 3.12
uv pip install -e .
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
