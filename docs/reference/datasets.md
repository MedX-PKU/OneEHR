# Dataset Converters

OneEHR includes built-in converters for standard clinical datasets. Each converter transforms raw source tables into OneEHR's three-table format (`dynamic.csv`, `static.csv`, `label.csv`).

## Usage

### CLI

```bash
oneehr convert --dataset mimic3 --raw-dir /path/to/mimic3 --output-dir data/mimic3/ --task mortality
```

### Python API

```python
from oneehr.datasets import MIMIC3Converter

converter = MIMIC3Converter("/path/to/mimic3")
result = converter.convert()

# Access DataFrames directly
print(result.dynamic.shape)
print(result.static.shape)
print(result.labels.keys())  # ['mortality', 'readmission', 'los_3day', 'los_7day']

# Or save to disk
converter.save("data/mimic3/", task="mortality")
```

---

## MIMIC-III

**Class:** `oneehr.datasets.MIMIC3Converter`

**Expected files:** `ADMISSIONS.csv`, `PATIENTS.csv`, `ICUSTAYS.csv`, `LABEVENTS.csv`, `CHARTEVENTS.csv`, `DIAGNOSES_ICD.csv`, `PROCEDURES_ICD.csv`, `PRESCRIPTIONS.csv`

**Event sources:**

| Source | Code prefix | Value |
|--------|-------------|-------|
| Lab events | `LAB_{itemid}` | Numeric result |
| Chart events | `CHART_{itemid}` | Numeric/text value |
| Diagnoses | `DX_{icd9_code}` | `1` (presence) |
| Procedures | `PROC_{icd9_code}` | `1` (presence) |
| Prescriptions | `RX_{drug}` | `1` (presence) |

**Static features:** `age`, `sex`, `ethnicity`, `insurance`

**Label tasks:**

| Task | Description |
|------|-------------|
| `mortality` | In-hospital mortality per admission |
| `readmission` | 30-day unplanned readmission |
| `los_3day` | Length of stay > 3 days |
| `los_7day` | Length of stay > 7 days |

**Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_chartevents` | `True` | Include chart events (large file) |
| `use_prescriptions` | `True` | Include prescriptions |
| `max_chartevents_rows` | `None` | Limit rows for memory control |

---

## MIMIC-IV

**Class:** `oneehr.datasets.MIMIC4Converter`

**Expected layout:** `hosp/` and `icu/` subdirectories (or flat directory fallback).

**Key differences from MIMIC-III:**
- Lowercase column names
- ICD version tracking (`icd_version` column: 9 or 10)
- Diagnosis codes prefixed with ICD version: `DX_ICD9_{code}` or `DX_ICD10_{code}`
- `anchor_age` field for direct age access
- `race` column in admissions (instead of `ethnicity`)

**Label tasks:** Same as MIMIC-III (mortality, readmission, los_3day, los_7day).

---

## eICU

**Class:** `oneehr.datasets.EICUConverter`

**Expected files:** `patient.csv`, `lab.csv`, `vitalPeriodic.csv`, `vitalAperiodic.csv`, `diagnosis.csv`, `medication.csv`

**Event sources:**

| Source | Code prefix | Value |
|--------|-------------|-------|
| Lab results | `LAB_{labname}` | Numeric result |
| Vital signs (periodic) | `VITAL_{column}` | Numeric value |
| Vital signs (aperiodic) | `VITAL_{column}` | Numeric value |
| Diagnoses | `DX_{icd9code}` | `1` |
| Medications | `RX_{drugname}` | `1` |

**Note:** eICU uses minute-based offsets from hospital admission. The converter creates synthetic timestamps for compatibility with OneEHR's time-based binning.

**Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_vitals` | `True` | Include vital sign events |
| `use_medication` | `True` | Include medication events |

---

## Custom Datasets

To convert any dataset, produce three CSVs matching the OneEHR data model:

```
dynamic.csv:  patient_id, event_time, code, value
static.csv:   patient_id, <covariates...>
label.csv:    patient_id, label_time, label_code, label_value
```

See the [Data Model](../getting-started/data-model.md) reference for column specifications.

You can also extend `BaseConverter`:

```python
from oneehr.datasets._base import BaseConverter, ConvertedDataset

class MyConverter(BaseConverter):
    def convert(self) -> ConvertedDataset:
        # Load and transform your data
        return ConvertedDataset(dynamic=..., static=..., labels={"mortality": ...})
```
