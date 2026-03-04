# Splits

All split strategies in OneEHR are **patient-level group splits**: no patient ever appears in more than one of train/val/test. This is fundamental to preventing data leakage in longitudinal EHR data.

---

## Split strategies

### K-fold cross-validation

Divide patients into k groups. Each fold uses one group as test, carves a validation set from the remaining patients, and trains on the rest.

```toml
[split]
kind = "kfold"
n_splits = 5
seed = 42
val_size = 0.2
```

| Parameter | Description |
|-----------|-------------|
| `n_splits` | Number of folds |
| `val_size` | Fraction of non-test patients used for validation |
| `seed` | Shuffle seed for reproducibility |

Use `fold_index` to run only a single fold (0-indexed):

```toml
[split]
kind = "kfold"
n_splits = 5
fold_index = 0   # run only the first fold
```

### Random split

A single train/val/test partition with configurable sizes.

```toml
[split]
kind = "random"
seed = 42
val_size = 0.1
test_size = 0.2
```

| Parameter | Description |
|-----------|-------------|
| `val_size` | Fraction of total patients for validation |
| `test_size` | Fraction of total patients for test |

### Time-based split

Prospective split: patients with all events before `time_boundary` go to the training pool; patients with events after go to test.

```toml
[split]
kind = "time"
time_boundary = "2012-01-01"
```

!!! note
    Even with a time boundary, the split is patient-level: patients are assigned entirely to one side based on their event history.

---

## Nested cross-validation

For time-based splits, you can run nested CV on the pre-boundary training pool. This gives multiple train/val folds within the temporal training set while preserving the prospective test set.

```toml
[split]
kind = "time"
time_boundary = "2012-01-01"
inner_kind = "kfold"
inner_n_splits = 5
```

This creates `inner_n_splits` folds within the pre-boundary patients, with the post-boundary patients held out as the prospective test set.

---

## How splits flow through the pipeline

1. `oneehr preprocess` materializes features (no splits yet)
2. `oneehr train` creates splits from the patient index, then for each split:
    - Masks train/val/test by patient_id
    - Fits the post-split pipeline on train only
    - Trains and evaluates the model
3. Split assignments are deterministic given the same `seed` and patient list

---

## Leakage prevention

OneEHR enforces patient-level isolation:

- A patient's **entire** event history belongs to one split
- Post-split pipeline operators (standardize, impute, etc.) are fit on the training set only
- Code vocabulary selection happens before splitting (on the full dataset) to ensure consistent feature schema
- Static features are encoded once, but postprocess transforms are fit per split

!!! warning "Don't bypass group splitting"
    If you implement custom data loading outside OneEHR, ensure your splits are patient-level. Random row-level splitting on longitudinal data causes severe temporal leakage.
