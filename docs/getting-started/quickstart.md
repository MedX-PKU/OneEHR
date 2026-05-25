# Quickstart

This quickstart runs the bundled TJH COVID-19 ICU example. It covers the standard command sequence: convert, preprocess, train, test, analyze, and plot.

Run the commands from the repository root.

## 1. Convert The Example Data

```bash
uv run python examples/tjh/convert.py
```

This writes the example CSV tables under `examples/tjh/`:

| File | Used For |
|------|----------|
| `dynamic.csv` | Longitudinal events shared by all tasks |
| `static.csv` | Patient-level covariates shared by all tasks |
| `label_mortality.csv` | Patient-level binary mortality |
| `label_mortality_time.csv` | Time-level binary mortality |
| `label_los.csv` | Time-level remaining length-of-stay regression |

## 2. Choose A Config

The example ships three ready-to-run configs:

| Config | Task | Prediction Mode | Models |
|--------|------|-----------------|--------|
| `examples/tjh/mortality_patient.toml` | Binary mortality | Patient | XGBoost, CatBoost, and DL models |
| `examples/tjh/mortality_time.toml` | Binary mortality | Time | XGBoost and GRU |
| `examples/tjh/los_time.toml` | Remaining LOS regression | Time | XGBoost and GRU |

Set the config path once:

```bash
CONFIG=examples/tjh/mortality_patient.toml
```

## 3. Preprocess

```bash
uv run oneehr preprocess --config "$CONFIG"
```

This creates `runs/tjh/preprocess/` with binned features, labels, static features when available, the patient split, and the run manifest.

## 4. Train

```bash
uv run oneehr train --config "$CONFIG"
```

This trains every model listed in `[[models]]` and writes checkpoints under `runs/tjh/train/`.

## 5. Test

```bash
uv run oneehr test --config "$CONFIG"
```

This evaluates trained models on the held-out test split and writes:

- `runs/tjh/test/predictions.parquet`
- `runs/tjh/test/metrics.json`

## 6. Analyze

```bash
uv run oneehr analyze --config "$CONFIG"
```

This writes JSON outputs under `runs/tjh/analyze/` for comparison, feature importance, fairness, calibration, statistical tests, and missing-data summaries.

## 7. Plot

```bash
uv run oneehr plot --config "$CONFIG" --style nature
```

Figures are written to `runs/tjh/figures/`. The plot command renders the figures whose required artifacts exist for the run.

## Run Another Example Task

Time-level mortality:

```bash
CONFIG=examples/tjh/mortality_time.toml
uv run oneehr preprocess --config "$CONFIG"
uv run oneehr train      --config "$CONFIG"
uv run oneehr test       --config "$CONFIG"
uv run oneehr analyze    --config "$CONFIG"
```

Time-level remaining length-of-stay regression:

```bash
CONFIG=examples/tjh/los_time.toml
uv run oneehr preprocess --config "$CONFIG"
uv run oneehr train      --config "$CONFIG"
uv run oneehr test       --config "$CONFIG"
uv run oneehr analyze    --config "$CONFIG"
```

## Use A Standard Dataset

Convert MIMIC-III, MIMIC-IV, or eICU before running the workflow:

```bash
oneehr convert --dataset mimic3 --raw-dir ~/data/mimic-iii --output-dir data/mimic3 --task mortality
```

Then point `[dataset]` in your TOML file to the converted CSVs. See [Dataset Converters](../reference/datasets.md) for dataset-specific layouts and tasks.
