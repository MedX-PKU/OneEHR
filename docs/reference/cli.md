# CLI Reference

OneEHR exposes six top-level commands:

- `preprocess` — bin features, split patients
- `train` — train ML/DL models
- `test` — evaluate on test set
- `analyze` — cross-system comparison and analysis
- `plot` — render publication-quality figures
- `convert` — convert raw datasets to OneEHR format

View the live interface with:

```bash
oneehr --help
```

## `oneehr preprocess`

```bash
oneehr preprocess --config <toml>
```

Bins dynamic events, generates labels, splits patients, and writes the run manifest. This is the required first step for every run.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |

Outputs are written to `{output.root}/{output.run_name}/preprocess/`.

## `oneehr train`

```bash
oneehr train --config <toml> [--force]
```

Trains all models defined in `[[models]]` and writes checkpoints under `{run_dir}/train/{model_name}/`.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--force` | No | Overwrite existing train directory |

## `oneehr test`

```bash
oneehr test --config <toml> [--force]
```

Runs all trained models and configured `[[systems]]` on the held-out test split. Writes `predictions.parquet` and `metrics.json` under `{run_dir}/test/`.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--force` | No | Overwrite existing test directory |

## `oneehr analyze`

```bash
oneehr analyze --config <toml> [--module NAME]
```

Reads `test/predictions.parquet` and writes structured analysis outputs under `{run_dir}/analyze/`.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--module` | No | Run a single analysis module instead of all |

Available modules:

- `comparison` — cross-system metrics comparison with bootstrap CI
- `feature_importance` — native importance for tree models, SHAP, permutation importance
- `fairness` — demographic parity, equalized odds, predictive parity, SMD
- `calibration` — temperature scaling, isotonic regression, ECE
- `statistical_tests` — DeLong, McNemar, BH FDR correction
- `missing_data` — missingness analysis per feature

## `oneehr plot`

```bash
oneehr plot --config <toml> [--figure NAME ...] [--style STYLE] [--output DIR]
```

Renders publication-quality figures from test/analyze results.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--figure` | No | Specific figure(s) to render (default: all available) |
| `--style` | No | Journal style preset: `default`, `nature`, `lancet`, `wide` |
| `--output` | No | Output directory for figures (default: `{run_dir}/figures/`) |

Available figures: `roc`, `pr`, `confusion`, `calibration`, `decision_curve`, `forest`, `fairness`, `training_curves`, `significance`, `missing_heatmap`, `cohort_flow`, `kaplan_meier`, `attribution`, `waterfall`.

## `oneehr convert`

```bash
oneehr convert --dataset <name> --raw-dir <path> --output-dir <path> [--task TASK]
```

Converts a raw clinical dataset into OneEHR's three-table format (`dynamic.csv`, `static.csv`, `label.csv`).

| Flag | Required | Description |
|------|:---:|-------------|
| `--dataset` | Yes | Source dataset: `mimic3`, `mimic4`, or `eicu` |
| `--raw-dir` | Yes | Path to raw dataset directory |
| `--output-dir` | Yes | Output directory for converted CSVs |
| `--task` | No | Label task to export (default: all tasks as separate files) |

### Supported datasets and tasks

| Dataset | Expected layout | Available tasks |
|---------|----------------|-----------------|
| `mimic3` | Flat directory with `ADMISSIONS.csv`, `PATIENTS.csv`, `LABEVENTS.csv`, etc. | `mortality`, `readmission`, `los_3day`, `los_7day` |
| `mimic4` | `hosp/` and `icu/` subdirectories | `mortality`, `readmission`, `los_3day`, `los_7day` |
| `eicu` | Flat directory with `patient.csv`, `lab.csv`, etc. | `mortality`, `los_3day`, `los_7day` |

Example:

```bash
# Convert MIMIC-III with mortality labels
oneehr convert --dataset mimic3 --raw-dir ~/data/mimic-iii/ --output-dir data/mimic3/ --task mortality

# Convert all available labels
oneehr convert --dataset mimic4 --raw-dir ~/data/mimic-iv/ --output-dir data/mimic4/
```
