# Data Model

OneEHR uses a three-table input specification. Users prepare plain CSV files -- there is no dataset registry or special format.

---

## Overview

| Table | Required | Description |
|-------|:---:|-------------|
| `dynamic.csv` | Yes | Longitudinal event table (long format) |
| `static.csv` | No | Patient-level static covariates |
| `label.csv` | No | Label events (task-agnostic long format) |

```toml
[dataset]
dynamic = "data/dynamic.csv"
static = "data/static.csv"     # optional
label = "data/label.csv"       # optional
```

---

## `dynamic.csv` (required)

The core input: a unified longitudinal event table in **long format** (one row per event).

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier |
| `event_time` | datetime | Timestamp (parseable by `pandas.to_datetime`) |
| `code` | string | Measurement or concept name (e.g. `"heart_rate"`, `"diagnosis_A01"`) |
| `value` | numeric or string | Measured value |

### Example

```csv
patient_id,event_time,code,value
P001,2023-01-01 08:00,heart_rate,72
P001,2023-01-01 08:00,blood_pressure_sys,120
P001,2023-01-01 08:00,diagnosis,A01
P001,2023-01-02 10:00,heart_rate,80
P002,2023-01-01 09:30,heart_rate,68
P002,2023-01-01 09:30,lab_glucose,5.4
```

### Key points

- Timestamps are **irregular** -- events for the same patient can occur at any time
- OneEHR bins events into fixed time windows during preprocessing (e.g. 1-hour, 1-day)
- `value` can be numeric or categorical; type inference happens automatically during binning (a code is considered numeric if 90%+ of its values parse as numbers)
- Numeric values: aggregated per bin via `preprocess.numeric_strategy` (mean or last)
- Categorical values: encoded per bin via `preprocess.categorical_strategy` (one-hot or count)

---

## `static.csv` (optional)

One row per patient with static covariates (demographics, baseline characteristics).

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier (must match `dynamic.csv`) |

All other columns are treated as static features. Numeric columns become `num__*` features; categorical columns become `cat__*__*` one-hot features.

### Example

```csv
patient_id,age,sex,insurance
P001,65,M,Medicare
P002,42,F,Private
P003,78,M,Medicaid
```

!!! tip
    Static features are concatenated with dynamic features for tabular models, and joined to test predictions for all models.

---

## `label.csv` (optional)

A task-agnostic label event table in long format. This allows multiple label types to coexist in one file.

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier |
| `label_time` | datetime | When the label was observed |
| `label_code` | string | Label type identifier (e.g. `"outcome"`, `"los"`) |
| `label_value` | numeric | Label value |

### Example

```csv
patient_id,label_time,label_code,label_value
P001,2023-01-05,outcome,1
P001,2023-01-05,los,4.5
P002,2023-01-03,outcome,0
P002,2023-01-03,los,2.0
```

---

## How tables flow through OneEHR

```
dynamic.csv ──┐
              ├──→ preprocess ──→ binned.parquet ──→ train
static.csv  ──┤                                       │
              │                                        ▼
label.csv ────┘                                    models + metrics
```

1. **Preprocess** bins `dynamic.csv` into fixed time windows and builds feature columns
2. Static features from `static.csv` are encoded into patient-level features
3. Labels come from `label.csv`
4. All downstream commands read from the materialized run artifacts instead of reinterpreting raw files
