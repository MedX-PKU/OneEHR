# OneEHR

**An all-in-one EHR predictive modeling and analysis library in Python.**

OneEHR is designed for researchers and clinicians who need to build predictive models from electronic health record (EHR) data. It provides an end-to-end pipeline from raw event tables to trained models with evaluation metrics.

---

## Key features

- **Doctor-friendly input** -- start from a single long-form event table (CSV/Excel)
- **Leakage prevention by default** -- all data splits are patient-level group splits
- **TOML-first experiments** -- most behavior is configured via one `experiment.toml`
- **CLI-first workflow** -- `preprocess` -> `train` -> `test` -> `analyze`
- **18 models** -- 5 tabular (XGBoost, CatBoost, RF, DT, GBDT) and 13 deep learning (GRU, LSTM, RNN, Transformer, TCN, MLP, AdaCare, StageNet, RETAIN, ConCare, GRASP, MCGRU, DrAgent)
- **Config-driven HPO** -- grid search built into the training pipeline
- **Probability calibration** -- temperature and Platt scaling for binary tasks

## The pipeline

```
               preprocess          train           test           analyze
dynamic.csv ──────────────→ features ────→ models ────→ metrics ────→ importance
static.csv  ──┘               │                │
label.csv   ──┘               ▼                ▼
                         run_manifest    summary.json
```

1. **Preprocess**: bin irregular events into fixed time windows, build features, materialize tabular views
2. **Train**: fit models with optional grid search, evaluate on patient-level splits
3. **Test**: evaluate trained models on external datasets
4. **Analyze**: compute feature importance (native, SHAP, attention)

## Quick start

```bash
# Install
uv venv .venv --python 3.12
uv pip install -e .

# Run the example experiment
uv run oneehr preprocess --config examples/experiment.toml
uv run oneehr train --config examples/experiment.toml
uv run oneehr analyze --config examples/experiment.toml
```

See the [Quickstart guide](getting-started/quickstart.md) for a detailed walkthrough.

## Documentation overview

| Section | What you'll find |
|---------|-----------------|
| [Getting Started](getting-started/installation.md) | Installation, quickstart tutorial, data model |
| [User Guide](guide/preprocessing.md) | In-depth guides for each pipeline stage |
| [Reference](reference/cli.md) | CLI flags, TOML parameters, model specs, artifact layout |
