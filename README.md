# OneEHR

An all-in-one EHR predictive modeling and analysis library in Python.

## Status

This repository is being rebuilt from scratch.

## CLI

Environment setup with uv:

```bash
uv venv .venv --python 3.12
uv pip install -e .
```

## Quickstart

```bash
oneehr preprocess --config examples/experiment.toml
oneehr train --config examples/experiment.toml
oneehr benchmark --config examples/experiment.toml
```



`oneehr benchmark` writes per-split metrics to `summary.csv`, and aggregated paper-style tables
including 95% CI to `paper_table.csv`.

When multiple models are configured, metrics are grouped by `model` in the output tables and
predictions/HPO artifacts are written under `logs/<run_name>/preds/<model>` and
`logs/<run_name>/hpo/<model>`.

## Models

You can configure one or more models in a single TOML:

```toml
[[models]]
name = "gru"

[[models]]
name = "xgboost"
```

Single-model config also works:

```toml
[model]
name = "rnn" # or "transformer"
```

### Labels (optional)

You can optionally generate labels via a Python function:

- Configure in `experiment.toml` under `[labels]`.
- Provide `fn = "path/to/label_fn.py:build_labels"`.

For N-N tasks (`task.prediction_mode = "time"`), your label function may return `label_time` and
OneEHR will internally align it to `bin_time` using `preprocess.bin_size`.

## Outputs

By default, run artifacts are written under `logs/`.

## Code Selection

Use `preprocess.code_selection` to control the code vocabulary:

- `frequency`: use `top_k_codes` by frequency (default)
- `all`: use all codes (set `top_k_codes = "all"` or omit)
- `list`: provide `code_list = ["A", "B"]`
- `importance`: load top-k from `importance_file` (e.g., SHAP importance)
