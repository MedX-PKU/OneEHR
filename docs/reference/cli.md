# CLI Reference

OneEHR exposes six top-level commands:

| Command | Purpose |
|---------|---------|
| `preprocess` | Bin event data, build features, create labels, and save the split contract |
| `train` | Train configured ML/DL models |
| `test` | Evaluate trained models and configured systems on the held-out test split |
| `analyze` | Write structured analysis outputs from `test/predictions.parquet` |
| `plot` | Render figures from saved run artifacts |
| `convert` | Convert supported raw datasets into OneEHR CSV tables |

View the live interface:

```bash
oneehr --help
```

## `oneehr preprocess`

```bash
oneehr preprocess --config <toml>
```

Reads the dataset tables, bins dynamic events, creates labels, saves the patient split, and writes the run manifest.

| Flag | Required | Description |
|------|:--------:|-------------|
| `--config` | Yes | Path to TOML experiment config |

Outputs are written to `{output.root}/{output.run_name}/preprocess/`.

## `oneehr train`

```bash
oneehr train --config <toml> [--force]
```

Trains all models defined in `[[models]]` and writes checkpoints under `{run_dir}/train/{model_name}/`.

| Flag | Required | Description |
|------|:--------:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--force` | No | Overwrite an existing train directory |

## `oneehr test`

```bash
oneehr test --config <toml> [--force]
```

Runs trained models and configured `[[systems]]` on the held-out test split. Writes `predictions.parquet` and `metrics.json` under `{run_dir}/test/`.

| Flag | Required | Description |
|------|:--------:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--force` | No | Overwrite an existing test directory |

## `oneehr analyze`

```bash
oneehr analyze --config <toml> [--module NAME]
```

Reads `test/predictions.parquet` and writes JSON outputs under `{run_dir}/analyze/`.

| Flag | Required | Description |
|------|:--------:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--module` | No | Run one analysis module instead of all modules |

Available modules:

| Module | Output |
|--------|--------|
| `comparison` | Metrics by system with bootstrap confidence intervals where supported |
| `feature_importance` | Native importance, SHAP, permutation importance, or integrated gradients depending on model type |
| `fairness` | Demographic parity, equalized odds, predictive parity, and SMD summaries for binary tasks |
| `calibration` | Calibration metrics and calibrated predictions for binary tasks |
| `statistical_tests` | Pairwise statistical tests and multiple-testing correction |
| `missing_data` | Missingness summaries from preprocessed features |

## `oneehr plot`

```bash
oneehr plot --config <toml> [--figure NAME ...] [--style STYLE] [--output DIR]
```

Renders figures whose required artifacts exist in the run directory.

| Flag | Required | Description |
|------|:--------:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--figure` | No | Figure names to render; defaults to all currently available figures |
| `--style` | No | Style preset: `default`, `nature`, `lancet`, or `wide` |
| `--output` | No | Output directory for figures; defaults to `{run_dir}/figures/` |

Registered figure names:

| Figure | Required Artifact |
|--------|-------------------|
| `roc` | `test/predictions.parquet` |
| `pr` | `test/predictions.parquet` |
| `forest` | `analyze/comparison.json` |
| `calibration` | `test/predictions.parquet` |
| `feature_importance` | `analyze/feature_importance.json` |
| `confusion` | `test/predictions.parquet` |
| `training_curves` | `train/` |
| `fairness` | `analyze/fairness.json` |
| `missing_data` | `preprocess/binned.parquet` |
| `decision_curve` | `test/predictions.parquet` |
| `significance` | `analyze/statistical_tests.json` |
| `cohort_flow` | `preprocess/split.json` |

## `oneehr convert`

```bash
oneehr convert --dataset <name> --raw-dir <path> --output-dir <path> [--task TASK]
```

Converts a supported raw clinical dataset into OneEHR CSV tables.

| Flag | Required | Description |
|------|:--------:|-------------|
| `--dataset` | Yes | Source dataset: `mimic3`, `mimic4`, or `eicu` |
| `--raw-dir` | Yes | Path to raw dataset directory |
| `--output-dir` | Yes | Output directory for converted CSVs |
| `--task` | No | Label task to export; defaults to all available tasks |

Supported datasets and tasks:

| Dataset | Expected Layout | Available Tasks |
|---------|-----------------|-----------------|
| `mimic3` | Flat directory with `ADMISSIONS.csv`, `PATIENTS.csv`, `LABEVENTS.csv`, and related MIMIC-III files | `mortality`, `readmission`, `los_3day`, `los_7day` |
| `mimic4` | `hosp/` and `icu/` subdirectories, with flat-directory fallback | `mortality`, `readmission`, `los_3day`, `los_7day` |
| `eicu` | Flat directory with `patient.csv`, `lab.csv`, and related eICU files | `mortality`, `los_3day`, `los_7day` |

Examples:

```bash
oneehr convert --dataset mimic3 --raw-dir ~/data/mimic-iii --output-dir data/mimic3 --task mortality
oneehr convert --dataset mimic4 --raw-dir ~/data/mimic-iv  --output-dir data/mimic4
```
