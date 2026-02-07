# OneEHR — Codex Agent Notes

This file provides persistent context for working on this repository with Codex.

## Project

- Name: **OneEHR**
- Positioning: **An all-in-one EHR predictive modeling and analysis library in Python**

## Tooling

- Use **uv** for environment and dependency management.
- Typical setup:

```bash
uv venv .venv --python 3.12
uv pip install -e .
```

- Python version: **3.12** (see `.python-version`).

## Architecture (pipeline)

OneEHR modules should follow the workflow:

1. Data Pre-processing
2. Modeling
3. Evaluation
4. Analysis

Future modules may include interpretability, uncertainty, confidence, etc.

### Data

- Input is doctor-friendly: start with **a single CSV/Excel table**.
- Default time column name: `event_time`.
- EHR data is **event-level** (irregular timestamps, `code`/`value`).
- Events can be **numeric and categorical**; handle typing/curation during preprocessing.
- Time discretization for deep learning uses **fixed-time binning** (user-configurable `bin_size`, e.g. `1h`, `1d`).

### Tasks

- Only **single-task** currently.
- Task types needed now: `binary` and `regression` (no multiclass yet).
- Prediction modes:
  - **N-1** (`patient`): one prediction per patient.
  - **N-N** (`time`): one prediction per time step.
- N-N labels can be generated via rules in a user-provided `label_fn`.

## Repository context

- Put longer-form architecture and design notes under `docs/`.

### Splits

- Must prevent leakage via **patient-level group split**.
- Support:
  - `k-fold`
  - `train/val/test`
  - `time-based split` (still patient-level grouping)

### CLI / Config

- CLI command name: `oneehr`.
- CLI should be driven primarily by **TOML configuration**.


## Git Commits

- git commit regularly with conventional commit message!
-