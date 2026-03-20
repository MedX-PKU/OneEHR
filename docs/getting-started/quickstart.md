# Quickstart

This quickstart uses the bundled TJH COVID-19 ICU example. It walks through the four core pipeline steps: preprocess, train, test, and analyze.

## 0. Convert The Source Data (once)

```bash
uv run python examples/tjh/convert.py
```

This reads the raw Excel file and produces inside `examples/tjh/`:

- `dynamic.csv`, `static.csv` -- shared across all tasks
- `label_mortality.csv` -- patient-level binary mortality
- `label_los.csv` -- time-level remaining LOS (regression)
- `label_mortality_time.csv` -- time-level binary mortality

## Example Configs

The TJH example ships three experiment configs:

| Config | Task | Mode | Models |
|--------|------|------|--------|
| `mortality_patient.toml` | Binary mortality | Patient (N-1) | All 6 |
| `mortality_time.toml` | Binary mortality | Time (N-N) | xgboost + gru |
| `los_time.toml` | Remaining LOS regression | Time (N-N) | xgboost + gru |

## 1. Preprocess

```bash
uv run oneehr preprocess --config examples/tjh/mortality_patient.toml
```

This creates `runs/tjh/preprocess/`, writes `manifest.json`, the split contract, and materializes binned feature views.

## 2. Train

```bash
uv run oneehr train --config examples/tjh/mortality_patient.toml
```

The default config trains 6 models (xgboost, catboost, gru, lstm, tcn, transformer) and writes checkpoints under `runs/tjh/train/`.

## 3. Test

```bash
uv run oneehr test --config examples/tjh/mortality_patient.toml
```

Evaluates all trained models on the held-out test split. Writes `runs/tjh/test/predictions.parquet` and `metrics.json`.

## 4. Analyze

```bash
uv run oneehr analyze --config examples/tjh/mortality_patient.toml
```

Writes structured analysis outputs under `runs/tjh/analyze/`, including cross-system comparison and feature importance.

## Try Other Tasks

Run the time-level mortality task:

```bash
uv run oneehr preprocess --config examples/tjh/mortality_time.toml
uv run oneehr train      --config examples/tjh/mortality_time.toml
uv run oneehr test       --config examples/tjh/mortality_time.toml
uv run oneehr analyze    --config examples/tjh/mortality_time.toml
```

Run the time-level remaining-LOS regression task:

```bash
uv run oneehr preprocess --config examples/tjh/los_time.toml
uv run oneehr train      --config examples/tjh/los_time.toml
uv run oneehr test       --config examples/tjh/los_time.toml
uv run oneehr analyze    --config examples/tjh/los_time.toml
```
