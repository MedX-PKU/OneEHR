# Label Functions

Label functions let you generate labels from event data using Python code. This is the recommended approach for rapid iteration -- you can change label definitions without modifying CSV files.

---

## Configuration

```toml
[labels]
fn = "examples/label_fn.py:build_labels"
bin_from_time_col = true
```

| Parameter | Description |
|-----------|-------------|
| `fn` | Python callable reference: `"path/to/file.py:function_name"` |
| `bin_from_time_col` | Floor `label_time` to bin boundaries using `preprocess.bin_size` |

---

## Function signature

Your label function receives four arguments and returns a DataFrame:

```python
def build_labels(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg,
) -> pd.DataFrame:
    ...
```

| Argument | Description |
|----------|-------------|
| `dynamic` | The raw dynamic event table (before binning) |
| `static` | The raw static table (or `None` if not provided) |
| `label` | The raw label table (or `None` if not provided) |
| `cfg` | The full `ExperimentConfig` object |

---

## Return format

The returned DataFrame must include specific columns depending on prediction mode:

### Patient-level (N-1)

| Column | Required | Description |
|--------|:---:|-------------|
| `patient_id` | Yes | Patient identifier |
| `label` | Yes | Label value |

### Time-level (N-N)

| Column | Required | Description |
|--------|:---:|-------------|
| `patient_id` | Yes | Patient identifier |
| `label_time` | Yes | Label timestamp (will be floored to bin if `bin_from_time_col = true`) |
| `label` | Yes | Label value |
| `mask` | Yes | 1 for valid labels, 0 for masked time steps |

---

## Example: binary outcome from label table

This example (from `examples/label_fn.py`) derives a binary outcome from a label event table:

```python
from __future__ import annotations
import pandas as pd


def build_labels(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    label: pd.DataFrame | None,
    cfg,
) -> pd.DataFrame:
    # Find each patient's last observed time
    df = dynamic[["patient_id", "event_time"]].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values(["patient_id", "event_time"])
    last_time = (
        df.groupby("patient_id", sort=False)
        .tail(1)
        .rename(columns={"event_time": "label_time"})
    )

    if label is not None and not label.empty:
        # Pick the label_code matching the task
        wanted = "outcome" if cfg.task.kind == "binary" else "los"
        lab = label.loc[label["label_code"] == wanted].copy()

        # For each patient, take the last label at/before last observed time
        merged = last_time.merge(lab, on="patient_id", how="left")
        merged = merged.loc[merged["label_time_y"] <= merged["label_time_x"]]
        merged = (
            merged.sort_values(["patient_id", "label_time_y"])
            .groupby("patient_id", sort=False)
            .tail(1)
        )
        out = merged.rename(
            columns={"label_value": "label", "label_time_x": "label_time"}
        )[["patient_id", "label_time", "label"]]
    else:
        out = last_time.copy()
        out["label"] = 1

    out["mask"] = 1
    return out[["patient_id", "label_time", "label", "mask"]]
```

---

## Tips

!!! tip "Version your label functions"
    Keep label function files next to your config. They are part of the experiment definition -- changing the label function changes the experiment.

!!! tip "Use `cfg` for task-aware labels"
    Access `cfg.task.kind` to branch between binary and regression labels, or `cfg.preprocess.bin_size` to align custom time logic.

!!! tip "N-N labels and bin alignment"
    For time-level prediction, set `bin_from_time_col = true` so OneEHR floors your `label_time` values to match the binned time grid. This ensures labels align with feature time steps.
