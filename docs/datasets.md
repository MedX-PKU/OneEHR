# Datasets in OneEHR (Decoupled, Three-Table Input)

OneEHR does **not** manage a registry of dataset names (e.g. TJH/MIMIC/etc).
Instead, OneEHR consumes a fixed, unified **three-table** input (standard CSV only):

- `dynamic.csv`: longitudinal event table (required)
- `static.csv`: patient-level table (optional)
- `label.csv`: label event table (optional, task-agnostic; tasks are defined by `label_fn`)

This keeps dataset-specific raw formats fully decoupled from the OneEHR framework.

## 1) `dynamic.csv` (Required, strict schema)

Unified longitudinal event table (long format). Required columns (fixed names):

- `patient_id`
- `event_time` — must be parseable by `pandas.to_datetime`
- `code` — concept / measurement name
- `value` — numeric or categorical value

This table is the only source for binning / sequence modeling.

## 2) `static.csv` (Optional, strict schema)

Patient-level table (one row per patient). Minimal requirement:

- `patient_id`

Other columns are treated as static covariates (e.g., sex, age).

If `dataset.static` is provided, OneEHR will automatically materialize the static
matrix and make it available to models that accept static covariates.

## 3) `label.csv` (Optional, Recommended, strict schema)

Task-agnostic label event table (long format). Required columns (fixed names):

- `patient_id`
- `label_time`
- `label_code`
- `label_value`

This format supports labels that can vary across visits/time.

## Recommended workflow: convert outside OneEHR

Write a script that reads raw files and exports these standard tables.
Then your experiment TOML only specifies paths:

```toml
[dataset.dynamic]
path = "/path/to/dynamic.csv"

[dataset.static]
path = "/path/to/static.csv"   # optional

[dataset.label]
path = "/path/to/label.csv"    # optional
```

## Labels: `label_fn` defines the task

OneEHR does not “register” tasks per dataset. Instead, the task is defined by `label_fn`,
which can combine multiple inputs:

```toml
[labels]
fn = "path/to/label_fn.py:build_labels"
```

Signature:

```python
def build_labels(dynamic, static, label, cfg) -> pd.DataFrame:
    ...
```

Return:
- patient mode (N-1): columns `patient_id`, `label`
- time mode (N-N): `patient_id`, plus `bin_time` or `label_time` (and optional `mask`)
