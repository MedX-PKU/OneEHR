# OneEHR

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/oneehr.svg)](https://pypi.org/project/oneehr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://medx-pku.github.io/OneEHR/)

OneEHR is a Python package for longitudinal EHR experiments. It uses standardized EHR tables and a TOML experiment config to run preprocessing, model training, testing, analysis, and figures across conventional ML/DL models and LLM or agent systems.

## What OneEHR Provides

- **Three-table EHR input contract**: `dynamic.csv`, optional `static.csv`, and optional `label.csv`
- **42 built-in model architectures**: tabular ML, recurrent and non-recurrent DL, irregular-time, multimodal, KG-enhanced, EHR-specialized, and survival models
- **One run directory per experiment**: preprocessing artifacts, checkpoints, predictions, metrics, analysis JSON, and figures
- **Dataset converters**: MIMIC-III, MIMIC-IV, and eICU converters for common clinical prediction tasks
- **Medical code utilities**: ICD-9/10 parsing, CCS grouping, ATC hierarchy, and code mapping helpers
- **Analysis modules**: comparison metrics, bootstrap confidence intervals, feature importance, fairness, calibration, statistical tests, and missing-data summaries
- **Plot commands**: ROC, PR, calibration, confusion, forest, fairness, missing-data, decision-curve, significance, cohort-flow, and training-curve figures

## Install

OneEHR requires Python 3.12+.

```bash
pip install oneehr
oneehr --help
```

Install from a source checkout:

```bash
git clone https://github.com/MedX-PKU/OneEHR.git
cd OneEHR
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"
uv run oneehr --help
```

Optional extras:

```bash
pip install "oneehr[survival]"  # lifelines support for survival analysis
pip install lime                # optional LIME interpretability support
```

## Quickstart

Run the bundled TJH COVID-19 ICU example from a source checkout:

```bash
uv run python examples/tjh/convert.py

CONFIG=examples/tjh/mortality_patient.toml
uv run oneehr preprocess --config "$CONFIG"
uv run oneehr train      --config "$CONFIG"
uv run oneehr test       --config "$CONFIG"
uv run oneehr analyze    --config "$CONFIG"
uv run oneehr plot       --config "$CONFIG" --style nature
```

The run is written to `runs/tjh/`:

```text
runs/tjh/
    manifest.json
    preprocess/
    train/
    test/
    analyze/
    figures/
```

Use the Python API with the config path:

```python
import oneehr

config_path = "examples/tjh/mortality_patient.toml"

preprocess_result = oneehr.preprocess(config_path)
train_result = oneehr.train(config_path)
test_result = oneehr.test(config_path)
analysis_result = oneehr.analyze(config_path)

print(preprocess_result.run_dir)
print(test_result.metrics_path)
print(analysis_result.modules_run)
```

## Input Data

OneEHR expects plain CSV files:

```text
dynamic.csv: patient_id, event_time, code, value
static.csv:  patient_id, <patient-level covariates...>
label.csv:   patient_id, label_time, label_code, label_value
```

Example TOML:

```toml
[dataset]
dynamic = "data/dynamic.csv"
static = "data/static.csv"
label = "data/label.csv"

[task]
kind = "binary"
prediction_mode = "patient"

[[models]]
name = "xgboost"

[[models]]
name = "gru"

[output]
root = "runs"
run_name = "my_experiment"
```

See [Data Model](docs/getting-started/data-model.md) and [Configuration](docs/reference/configuration.md) for the full contract.

## CLI Workflow

```bash
oneehr preprocess --config experiment.toml
oneehr train      --config experiment.toml
oneehr test       --config experiment.toml
oneehr analyze    --config experiment.toml
oneehr plot       --config experiment.toml
```

Dataset conversion:

```bash
oneehr convert --dataset mimic3 --raw-dir /path/to/mimic3 --output-dir data/mimic3 --task mortality
oneehr convert --dataset mimic4 --raw-dir /path/to/mimic4 --output-dir data/mimic4 --task mortality
oneehr convert --dataset eicu   --raw-dir /path/to/eicu   --output-dir data/eicu   --task mortality
```

## Models

| Category | Config names |
|----------|--------------|
| Tabular ML | `xgboost`, `catboost`, `rf`, `dt`, `gbdt`, `lr` |
| Recurrent | `gru`, `lstm`, `rnn`, `grud`, `dipole`, `hitanet`, `m3care`, `pai` |
| Non-recurrent | `cnn`, `tcn`, `transformer`, `sand`, `mlp`, `deepr`, `mamba`, `jamba`, `lsan` |
| Irregular-time | `mtand`, `raindrop`, `contiformer`, `teco` |
| EHR-specialized | `adacare`, `stagenet`, `retain`, `concare`, `grasp`, `mcgru`, `dragent`, `prism`, `safari` |
| Multimodal | `emerge` |
| KG-enhanced | `graphcare`, `kerprint`, `protoehr` |
| Survival | `deepsurv`, `deephit` |

Models with static branches use patient-level static features when `static.csv` is provided. KG-enhanced models use the built-in `lightweight_auto` KG preset unless `kg_source = "external"` and `external_kg_path` are provided.

## Documentation

Full documentation: [medx-pku.github.io/OneEHR](https://medx-pku.github.io/OneEHR/)

- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [Data Model](docs/getting-started/data-model.md)
- [Core Workflows](docs/guide/core-workflows.md)
- [CLI Reference](docs/reference/cli.md)
- [Configuration](docs/reference/configuration.md)
- [Models](docs/reference/models.md)
- [Artifacts](docs/reference/artifacts.md)
- [Dataset Converters](docs/reference/datasets.md)
- [Medical Codes](docs/reference/medcode.md)

Run the documentation site locally:

```bash
uv run --group docs mkdocs serve
```

Build the static site:

```bash
uv run --group docs mkdocs build
```

## Development

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[test]"
pytest tests/ -v
ruff check oneehr tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
