# Quickstart

This quickstart uses the bundled TJH COVID-19 ICU example. It walks through the core pipeline steps: preprocess, train, test, analyze, and plot.

For more detailed tutorials with Jupyter notebooks, see [Tutorials](../tutorials.md).

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
| `mortality_patient.toml` | Binary mortality | Patient (N-1) | 25 DL + tabular |
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

Trains all models configured in `[[models]]` and writes checkpoints under `runs/tjh/train/`.

## 3. Test

```bash
uv run oneehr test --config examples/tjh/mortality_patient.toml
```

Evaluates all trained models on the held-out test split. Writes `runs/tjh/test/predictions.parquet` and `metrics.json`.

## 4. Analyze

```bash
uv run oneehr analyze --config examples/tjh/mortality_patient.toml
```

Writes structured analysis outputs under `runs/tjh/analyze/`, including cross-system comparison, feature importance, fairness metrics, and statistical tests.

## 5. Plot (Optional)

```bash
oneehr plot --config examples/tjh/mortality_patient.toml --style nature
```

Renders publication-quality figures (ROC, PR, calibration, forest plot, etc.) under `runs/tjh/figures/`.

## Using Standard Datasets

Convert MIMIC-III, MIMIC-IV, or eICU data before running the pipeline:

```bash
oneehr convert --dataset mimic3 --raw-dir ~/data/mimic-iii/ --output-dir data/mimic3/ --task mortality
```

Then update your TOML config to point to the converted files. See [Dataset Converters](../reference/datasets.md) for details.

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
