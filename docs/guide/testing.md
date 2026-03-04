# Testing

`oneehr test` evaluates trained models on an external test dataset. It reads the training run's manifest and applies the same feature schema to new data.

---

## Basic usage

```bash
uv run oneehr test --config experiment.toml
```

### With explicit run directory

```bash
uv run oneehr test --config experiment.toml --run-dir logs/my_run
```

### With external test dataset

```bash
uv run oneehr test --config experiment.toml --test-dataset path/to/test_config.toml
```

---

## How test evaluation works

1. **Read the run manifest** from the training run directory
2. **Load test data** from one of (in priority order):
    - `--test-dataset` CLI flag
    - `[datasets.test]` config section
    - `[dataset]` config section (same data as training)
3. **Re-bin test events** using the training-time code vocabulary (`code_selection = "list"` with the exact codes from training)
4. **Align feature columns** to match the training schema (missing columns filled with 0, extra columns dropped)
5. **Re-apply static feature postprocessing** from the manifest
6. **For each trained model/split**:
    - Load the model checkpoint
    - Generate predictions on test data
    - Compute metrics
7. **Write results** to `test_runs/` (or `--out-dir`)

---

## Test dataset configuration

The test dataset follows the same three-table format as training:

```toml
[dataset]
dynamic = "data/test_dynamic.csv"
static = "data/test_static.csv"
label = "data/test_label.csv"
```

!!! important "Feature alignment"
    The test pipeline automatically aligns feature columns to the training schema. You don't need to ensure the test dataset has exactly the same codes -- missing codes are filled with zeros, and extra codes are ignored.

---

## Output files

| File | Location | Description |
|------|----------|-------------|
| `metrics.json` | `test_runs/{model}/{split}/` | Per-split test metrics |
| `preds.parquet` | `test_runs/{model}/{split}/` | Test predictions |
| `test_summary.json` | Run root | Aggregated test summary |

---

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *required* | Path to TOML config |
| `--run-dir` | from config | Override the training run directory |
| `--test-dataset` | `None` | Override the test dataset config |
| `--force` | `false` | Overwrite existing test results |
| `--out-dir` | `None` | Custom output directory for results |
