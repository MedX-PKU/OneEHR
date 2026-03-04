# Testing

`oneehr test` evaluates trained models on a test dataset. By default, it evaluates the model on the held-out test patients defined by the splits generated during training. It can also evaluate on completely external test datasets.

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
2. **Determine test mode**:
    - **Self-split mode (default)**: Uses the exact train/val/test splits saved in `run_root/splits/` and filters the preprocessed training artifacts to evaluate ONLY on the held-out test patients. This guarantees no data leakage.
    - **External mode**: If `--test-dataset` is provided (or `[datasets.test]`), loads the external data, re-bins events using the training-time code vocabulary, and aligns feature columns to match the training schema.
3. **Re-apply static feature postprocessing** from the manifest or load the split-specific fitted postprocess pipeline.
4. **For each trained model/split**:
    - Load the model checkpoint
    - Generate predictions on the test set
    - Compute metrics
5. **Write results** to `test_runs/` (or `--out-dir`)

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
