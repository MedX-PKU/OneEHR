# OneEHR

OneEHR is a Python platform for longitudinal EHR experiments. It provides shared infrastructure for preprocessing, modeling, testing, and analysis across conventional ML/DL models and LLM systems on one shared run contract.

## Workflow At A Glance

```bash
uv run oneehr preprocess --config experiment.toml
uv run oneehr train      --config experiment.toml
uv run oneehr test       --config experiment.toml
uv run oneehr analyze    --config experiment.toml
```

All four commands operate on the same run directory under `{output.root}/{output.run_name}/`.

## Install

OneEHR requires Python 3.12 and `uv`.

```bash
uv venv .venv --python 3.12
uv pip install -e .
uv run oneehr --help
```

Optional docs extras:

```bash
uv pip install -e ".[docs]"
```

## Quickstart

Use the bundled TJH example at [`examples/tjh/mortality_patient.toml`](examples/tjh/mortality_patient.toml):

```bash
# Convert source data (only needed once)
uv run python examples/tjh/convert.py

# Run the pipeline
uv run oneehr preprocess --config examples/tjh/mortality_patient.toml
uv run oneehr train      --config examples/tjh/mortality_patient.toml
uv run oneehr test       --config examples/tjh/mortality_patient.toml
uv run oneehr analyze    --config examples/tjh/mortality_patient.toml
```

This writes the run under `runs/tjh/`, including `manifest.json`, `preprocess/`, `train/`, `test/`, and `analyze/`.

## Models

OneEHR ships a growing model zoo across tabular and deep learning baselines.

**Tabular:** XGBoost, CatBoost, Random Forest, Decision Tree, GBDT, Logistic Regression

**Recurrent / sequence:** GRU, LSTM, RNN, M3Care, PAI (GRU + pseudo-imputation plugin)

**Non-recurrent:** TCN, Transformer, MLP, Deepr, EHR-Mamba, Jamba

**EHR-specialised:** AdaCare, StageNet, RETAIN, ConCare, GRASP, MCGRU, DrAgent, PRISM, SAFARI

Models with static branches (ConCare, GRASP, MCGRU, DrAgent, PRISM, SAFARI) automatically use patient-level static features when `static.csv` is provided.

## Configuration Model

OneEHR uses TOML as the experiment contract. The main sections are:

- `[dataset]` for input table paths (`dynamic`, `static`, `label`)
- `[preprocess]` for binning and feature building
- `[task]` for task kind and prediction mode
- `[split]` for patient-level train/val/test splitting
- `[[models]]` for model selection with per-model `params`
- `[trainer]` for DL training configuration
- `[[systems]]` for LLM/agent system definitions
- `[output]` for run root and run name

The standard input model is:

- `dynamic.csv` required: long-form event table with `patient_id`, `event_time`, `code`, and `value`
- `static.csv` optional: patient-level covariates keyed by `patient_id`
- `label.csv` optional: label events keyed by `patient_id` and `label_time`

Prediction modes:

- `patient`: patient-level N-1 prediction
- `time`: time-window N-N prediction

## Documentation

- [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
- [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md)
- [`docs/getting-started/data-model.md`](docs/getting-started/data-model.md)
- [`docs/guide/core-workflows.md`](docs/guide/core-workflows.md)
- [`docs/reference/cli.md`](docs/reference/cli.md)
- [`docs/reference/configuration.md`](docs/reference/configuration.md)
- [`docs/reference/models.md`](docs/reference/models.md)
- [`docs/reference/artifacts.md`](docs/reference/artifacts.md)

Build the docs locally:

```bash
uv pip install -e ".[docs]"
uv run mkdocs serve
uv run mkdocs build
```

## Validation

```bash
uv run oneehr --help
uv run pytest tests/ -v
uv run oneehr preprocess --config examples/tjh/mortality_patient.toml
uv run mkdocs build
```
