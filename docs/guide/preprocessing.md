# Preprocessing

Preprocessing converts irregular EHR events into fixed-time binned features suitable for modeling. This is the first step in every OneEHR experiment.

---

## How binning works

Raw EHR events have irregular timestamps -- multiple measurements at different times for each patient. OneEHR bins these into fixed time windows:

1. Each event's `event_time` is floored to the nearest bin boundary (e.g. daily, hourly)
2. Events within the same bin are aggregated per code
3. The result is a regular grid: one row per (patient, bin_time) with feature columns

```
Raw events:                          Binned (1-day):
P001, 08:15, heart_rate, 72         P001, 2023-01-01, num__heart_rate=76
P001, 14:30, heart_rate, 80    →    P001, 2023-01-02, num__heart_rate=85
P001, 14:30, diagnosis, A01         P001, 2023-01-01, cat__diagnosis__A01=1
P001, 09:00, heart_rate, 85
```

## Bin size

The `bin_size` parameter controls the granularity:

```toml
[preprocess]
bin_size = "1d"   # 1 day (default)
# bin_size = "1h"   # 1 hour
# bin_size = "6h"   # 6 hours
```

Smaller bins capture finer temporal patterns but create longer sequences and sparser features. For most clinical tasks, `"1d"` is a good starting point.

## Aggregation strategies

### Numeric values

When multiple numeric values fall in the same bin for the same code:

| Strategy | Description |
|----------|-------------|
| `mean` | Average of all values in the bin (default) |
| `last` | Last observed value in the bin |

```toml
[preprocess]
numeric_strategy = "mean"
```

### Categorical values

When a code has categorical values:

| Strategy | Description |
|----------|-------------|
| `onehot` | Each unique (code, value) pair becomes a binary column `cat__{code}__{value}` (default) |
| `count` | Per-bin count of occurrences as `cat__{code}` |

```toml
[preprocess]
categorical_strategy = "onehot"
```

!!! note "Type inference"
    OneEHR automatically determines whether a code is numeric or categorical. A code is numeric if 90%+ of its values parse as numbers; otherwise it's categorical.

---

## Code selection

Not all codes in your dataset may be useful. `code_selection` controls which codes enter the feature space:

=== "Frequency (default)"

    Select the top-k most frequent codes:

    ```toml
    [preprocess]
    code_selection = "frequency"
    top_k_codes = 500
    min_code_count = 1
    ```

=== "All"

    Include every code:

    ```toml
    [preprocess]
    code_selection = "all"
    ```

=== "Explicit list"

    Specify codes manually:

    ```toml
    [preprocess]
    code_selection = "list"
    code_list = ["heart_rate", "blood_pressure_sys", "lab_glucose"]
    ```

=== "Importance"

    Load codes ranked by importance (e.g. from a previous SHAP analysis):

    ```toml
    [preprocess]
    code_selection = "importance"
    top_k_codes = 100
    importance_file = "results/shap_importance.csv"
    importance_code_col = "code"
    importance_value_col = "importance"
    ```

---

## Post-split pipeline

The `pipeline` applies transformations to the feature matrix **after** splitting (fit on train only, transform all splits). Operators run in order.

```toml
[preprocess]
pipeline = [
  { op = "standardize", cols = "num__*" },
  { op = "impute", strategy = "mean", cols = "num__*" },
]
```

### Available operators

#### `standardize`

Z-score normalization: `(x - mean) / std`. Fit on training set.

```toml
{ op = "standardize", cols = "num__*" }
```

#### `impute`

Fill missing values. Strategies: `mean`, `median`, `mode`, `constant`.

```toml
{ op = "impute", strategy = "mean", cols = "num__*" }
```

#### `forward_fill`

Within-patient temporal forward fill, with fallback imputation for leading NaNs.

```toml
{ op = "forward_fill", cols = "num__*" }
```

#### `clip`

Hard clip values to bounds.

```toml
{ op = "clip", cols = "num__*", lower = 0.0, upper = 300.0 }
```

#### `winsorize`

Quantile-based clipping. Boundaries are fit on the training set.

```toml
{ op = "winsorize", cols = "num__*", lower_q = 0.01, upper_q = 0.99 }
```

!!! tip "Column patterns"
    The `cols` parameter supports glob patterns. Use `"num__*"` for all numeric features, `"cat__*"` for all categorical features, or specific names like `"num__heart_rate"`.

---

## Feature naming convention

After binning, feature columns follow a naming convention:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `num__` | Numeric feature | `num__heart_rate`, `num__lab_glucose` |
| `cat__` | Categorical feature | `cat__diagnosis__A01`, `cat__sex__M` |

This convention is consistent across dynamic and static features.
