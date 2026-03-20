# AGENTS.md

## Project Snapshot

- **Project name:** OneEHR
- **Purpose:** CLI tool for longitudinal EHR data processing, modeling, and evaluation
- **CLI:** `oneehr`
- **Configuration style:** TOML
- **Canonical wording:** `docs/reference/positioning.md`
- **Python version:** 3.12 (`.python-version`)
- **Dependency manager:** `uv` (required)

## Package Structure

```
oneehr/
    __init__.py
    cli/                    # 5 commands: preprocess, train, test, eval, analyze
    config/                 # schema.py (~18 dataclasses), load.py (TOML loader)
    data/                   # io, binning, tabular (views + postprocess + features), splits, sequence
    models/                 # __init__.py (registry), recurrent, transformer, tcn, tree
    training/               # trainer, hpo, persistence
    eval/                   # metrics, bootstrap, calibration (temperature only), workflow, query, tables
    agent/                  # client, runtime, contracts, schema, templates, workflow
    analysis/               # feature_importance, reporting (includes readers)
    artifacts/              # manifest, store, materialize
    utils/                  # IO, imports, time (all in __init__.py)
```

## Supported Models

- **DL:** GRU, LSTM (unified in recurrent.py), Transformer, TCN
- **ML:** XGBoost, CatBoost

## CLI Surface

```
oneehr preprocess --config experiment.toml [--overview]
oneehr train      --config experiment.toml [--force]
oneehr test       --config experiment.toml [--run-dir DIR] [--force]
oneehr eval       build|run|report|trace|instance --config experiment.toml
oneehr analyze    --config experiment.toml [--module NAME] [--method shap|xgboost]
```

## Adding a New Model

1. Create `models/my_model.py`
2. In `models/__init__.py`, call `register("my_model", ModelSpec(...))`
3. Add config dataclass in `config/schema.py` and field in `ModelConfig`
