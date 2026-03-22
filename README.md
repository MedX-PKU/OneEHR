# OneEHR

[![CI](https://github.com/MedXLab/OneEHR/actions/workflows/ci.yml/badge.svg)](https://github.com/MedXLab/OneEHR/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://medxlab.github.io/OneEHR)

OneEHR is a unified Python platform for longitudinal EHR experiments across ML, DL, and LLM agents. It provides shared infrastructure for preprocessing, modeling, testing, and analysis on one shared run contract — the first toolkit bridging classical machine learning, deep learning, and agentic AI for clinical prediction.

## Key Features

- **38 model architectures** — tabular ML, recurrent/non-recurrent DL, irregular-time, KG-enhanced, and survival models
- **Unified ML/DL/LLM comparison** — all predictions in one `predictions.parquet` with bootstrap CI and statistical tests
- **Dataset converters** — built-in support for MIMIC-III, MIMIC-IV, and eICU
- **Medical code ontologies** — ICD-9/10 mapping, CCS grouping, ATC drug hierarchy
- **Survival analysis** — DeepSurv, DeepHit, concordance index, Kaplan-Meier visualization
- **Fairness & interpretability** — demographic parity, equalized odds, SHAP, LIME, integrated gradients, attention visualization
- **Publication-quality figures** — ROC, PR, calibration, DCA, forest plots, KM curves with Nature/Lancet style presets
- **Reproducibility by design** — single TOML config = complete experiment specification

## Workflow At A Glance

```bash
oneehr preprocess --config experiment.toml   # Bin features, split patients
oneehr train      --config experiment.toml   # Train ML/DL models
oneehr test       --config experiment.toml   # Evaluate on test set
oneehr analyze    --config experiment.toml   # Cross-system comparison
oneehr plot       --config experiment.toml   # Publication figures
```

All commands operate on the same run directory under `{output.root}/{output.run_name}/`.

## Install

OneEHR requires Python 3.12+.

```bash
pip install oneehr

# Or from source:
uv venv .venv --python 3.12
uv pip install -e .
oneehr --help
```

## Quickstart

Use the bundled TJH COVID-19 ICU example:

```bash
# Convert source data (only needed once)
python examples/tjh/convert.py

# Run the full pipeline
oneehr preprocess --config examples/tjh/mortality_patient.toml
oneehr train      --config examples/tjh/mortality_patient.toml
oneehr test       --config examples/tjh/mortality_patient.toml
oneehr analyze    --config examples/tjh/mortality_patient.toml
```

Or use the Python API:

```python
import oneehr

config = oneehr.load_config("examples/tjh/mortality_patient.toml")
oneehr.preprocess(config)
oneehr.train(config)
oneehr.test(config)
oneehr.analyze(config)
```

## Dataset Converters

Convert standard clinical datasets into OneEHR's three-table format:

```bash
# MIMIC-III
oneehr convert --dataset mimic3 --raw-dir /path/to/mimic3 --output-dir data/mimic3/ --task mortality

# MIMIC-IV
oneehr convert --dataset mimic4 --raw-dir /path/to/mimic4 --output-dir data/mimic4/ --task mortality

# eICU
oneehr convert --dataset eicu --raw-dir /path/to/eicu --output-dir data/eicu/ --task mortality
```

Each converter produces labels for mortality, readmission, and length-of-stay tasks.

## Models

OneEHR ships 38 model architectures:

| Category | Models |
|----------|--------|
| **Tabular ML** | XGBoost, CatBoost, Random Forest, Decision Tree, GBDT, Logistic Regression |
| **Recurrent** | GRU, LSTM, RNN, GRU-D, Dipole, HiTANet, M3Care, PAI |
| **Non-recurrent** | CNN, TCN, Transformer, SAnD, MLP, Deepr, EHR-Mamba, Jamba, LSAN |
| **Irregular-time** | mTAND, Raindrop, ContiFormer, TECO |
| **EHR-specialised** | AdaCare, StageNet, RETAIN, ConCare, GRASP, MCGRU, DrAgent, PRISM, SAFARI |
| **KG-enhanced** | GraphCare, KerPrint, ProtoEHR |
| **Survival** | DeepSurv, DeepHit |

Models with static branches (ConCare, GRASP, MCGRU, DrAgent, PRISM, SAFARI, TECO) automatically use patient-level static features when `static.csv` is provided.

## Task Types

| Task | Config | Description |
|------|--------|-------------|
| Binary classification | `kind = "binary"` | Mortality, readmission, etc. |
| Multiclass | `kind = "multiclass"` | Phenotyping, diagnosis groups |
| Regression | `kind = "regression"` | Length of stay, lab value prediction |
| Survival | `kind = "survival"` | Time-to-event with censoring |
| Multi-label | `kind = "multilabel"` | ICD coding, multi-diagnosis |

## Medical Code Ontologies

```python
from oneehr.medcode import ICD9, ICD10, CodeMapper, CCSGrouper, ATCHierarchy

# ICD code utilities
ICD9.chapter("401.9")    # → "Circulatory system"
ICD10.category("I10.0")  # → "I10"

# Aggregate codes by ontology for dimensionality reduction
mapper = CodeMapper()
mapper.add_icd_chapter_mapping(version=9)
mapped_events = mapper.apply(events_df)
```

## Configuration

OneEHR uses TOML as the experiment contract:

- `[dataset]` — input table paths (`dynamic`, `static`, `label`)
- `[preprocess]` — binning, feature engineering, preprocessing pipeline
- `[task]` — task kind and prediction mode (`patient` or `time`)
- `[split]` — patient-level train/val/test splitting
- `[[models]]` — model selection with per-model `params`
- `[trainer]` — DL training config (mixed precision, LR schedulers, early stopping)
- `[[systems]]` — LLM/agent system definitions
- `[output]` — run root and run name

## Tutorials

| Tutorial | Description |
|----------|-------------|
| [01 Quickstart](tutorials/01_quickstart.ipynb) | End-to-end TJH mortality prediction |
| [02 Custom Dataset](tutorials/02_custom_dataset.ipynb) | Bring your own data + medical code mapping |
| [03 Model Comparison](tutorials/03_model_comparison.ipynb) | ML vs DL with bootstrap CI and statistical tests |
| [04 Fairness & Explainability](tutorials/04_fairness_analysis.ipynb) | Bias detection + feature importance |
| [05 Survival Analysis](tutorials/05_survival_analysis.ipynb) | DeepSurv, C-index, Kaplan-Meier curves |

## Documentation

Full documentation: [medxlab.github.io/OneEHR](https://medxlab.github.io/OneEHR)

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

Build docs locally:

```bash
uv pip install -e ".[docs]"
uv run mkdocs serve
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Validation

```bash
pytest tests/ -v                                                    # 114 tests
oneehr preprocess --config examples/tjh/mortality_patient.toml      # End-to-end
```
