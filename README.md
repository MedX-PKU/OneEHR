# OneEHR

An all-in-one EHR predictive modeling and analysis library in Python.

## Status

This repository is being rebuilt from scratch.

## CLI

Environment setup with uv:

```bash
uv venv .venv --python 3.12
uv sync
```

Add dependencies as needed:

```bash
uv add torch
uv add xgboost scikit-learn
```

## Quickstart (MVP: XGBoost)

```bash
oneehr preprocess --config examples/experiment.toml
oneehr train --config examples/experiment.toml
oneehr benchmark --config examples/experiment.toml
```

## Multi-GPU (single node)

For deep learning models (GRU/RNN/Transformer), you can enable single-node multi-GPU training via DDP
in your TOML:

```toml
[trainer]
ddp = true
ddp_backend = "nccl"
```

Then run:

```bash
oneehr benchmark --config examples/experiment.toml
```

Notes:

- DDP runs via `torch.multiprocessing.spawn` (one process per GPU).
- Metrics/predictions are written by rank 0 only.

`oneehr benchmark` writes per-split metrics to `summary.csv`, and aggregated paper-style tables
including 95% CI to `paper_table.csv`.

## GRU (baseline)

For a quick GRU baseline, set in TOML:

```toml
[model]
name = "gru"
```

## RNN / Transformer

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
