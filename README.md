# OneEHR

An all-in-one EHR predictive modeling and analysis library in Python.

## Status

This repository is being rebuilt from scratch.

## CLI

After installing (editable recommended) with uv:

```bash
uv venv .venv --python 3.12
uv pip install -e .
uv pip install -e ".[ml]"
oneehr --help
```

## Quickstart (MVP: XGBoost)

```bash
oneehr preprocess --config examples/experiment.toml
oneehr train --config examples/experiment.toml
oneehr benchmark --config examples/experiment.toml
```

## GRU (baseline)

For a quick GRU baseline, set in TOML:

```toml
[model]
name = "gru"
```

### Labels (optional)

You can optionally generate labels via a Python function:

- Configure in `experiment.toml` under `[labels]`.
- Provide `fn = "path/to/label_fn.py:build_labels"`.

For N-N tasks (`task.prediction_mode = "time"`), your label function may return `label_time` and
OneEHR will internally align it to `bin_time` using `preprocess.bin_size`.
