# Data Model

OneEHR reads plain CSV files. The core input is a longitudinal event table; static covariates and labels are separate tables keyed by patient.

## Overview

| Table | Required | Purpose |
|-------|:--------:|---------|
| `dynamic.csv` | Yes | Longitudinal event table in long format |
| `static.csv` | No | Patient-level covariates such as demographics or baseline features |
| `label.csv` | No | Label events for one or more prediction tasks |

```toml
[dataset]
dynamic = "data/dynamic.csv"
static = "data/static.csv"
label = "data/label.csv"
```

## `dynamic.csv`

`dynamic.csv` is one row per observed event.

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier |
| `event_time` | datetime | Timestamp parseable by `pandas.to_datetime` |
| `code` | string | Measurement, diagnosis, procedure, medication, or feature name |
| `value` | numeric or string | Observed value |

Example:

```csv
patient_id,event_time,code,value
P001,2023-01-01 08:00,heart_rate,72
P001,2023-01-01 08:00,blood_pressure_sys,120
P001,2023-01-01 08:00,diagnosis,A01
P001,2023-01-02 10:00,heart_rate,80
P002,2023-01-01 09:30,heart_rate,68
P002,2023-01-01 09:30,lab_glucose,5.4
```

Preprocessing bins events into fixed time windows such as `1h`, `6h`, or `1d`. Numeric event values are aggregated with `preprocess.numeric_strategy`. Categorical values are encoded with `preprocess.categorical_strategy`.

## `static.csv`

`static.csv` has one row per patient.

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier matching `dynamic.csv` |
| Other columns | numeric or string | Static covariates |

Example:

```csv
patient_id,age,sex,insurance
P001,65,M,Medicare
P002,42,F,Private
P003,78,M,Medicaid
```

Numeric static columns become `num__*` features. Categorical static columns become `cat__*__*` one-hot features. Models with static branches receive these features as a separate tensor; tabular models receive them in the flattened feature matrix.

## `label.csv`

`label.csv` is a long-format label table. A single file can contain multiple label types through `label_code`.

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | Patient identifier |
| `label_time` | datetime | When the label is observed |
| `label_code` | string | Label type, for example `outcome` or `los` |
| `label_value` | numeric | Label value |

Example:

```csv
patient_id,label_time,label_code,label_value
P001,2023-01-05,outcome,1
P001,2023-01-05,los,4.5
P002,2023-01-03,outcome,0
P002,2023-01-03,los,2.0
```

Patient-level tasks use one label per patient. Time-level tasks align labels to binned time windows.

## Flow Through A Run

```text
dynamic.csv --+
              +--> preprocess --> binned features + labels --> train --> test --> analyze
static.csv  --+
label.csv   --+
```

Downstream commands read the saved run artifacts instead of reading the raw CSV files again. This keeps training, testing, analysis, and plotting tied to the same preprocessed data and split.
