# OneEHR Project Context

This folder stores project-specific context used to guide development with Codex.

## Positioning

OneEHR: **An all-in-one EHR predictive modeling and analysis library in Python**.

## Key requirements (current)

- Input is a single doctor-friendly CSV/Excel event table.
- Default time column: `event_time`.
- Fixed-time binning (user-configurable `bin_size` like `1h`, `1d`).
- Values can be numeric or categorical.
- Labels:
  - May be included in the input table.
  - May be generated via `[labels]` `label_fn`.
- Tasks:
  - Single-task only for now.
  - Binary classification and regression.
  - Prediction modes: N-1 (`patient`) and N-N (`time`).
- Splits:
  - Must be patient-level group split.
  - Support k-fold / train-val-test / time-based.

