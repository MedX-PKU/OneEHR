# Quickstart

This quickstart uses the bundled TJH COVID-19 ICU example at `examples/tjh/experiment.toml`. It walks through the four core pipeline steps: preprocess, train, test, and analyze.

## 0. Convert The Source Data (once)

```bash
uv run python examples/tjh/convert.py
```

This reads the raw Excel file and produces `dynamic.csv`, `static.csv`, and `label.csv` inside `examples/tjh/`.

## 1. Preprocess

```bash
uv run oneehr preprocess --config examples/tjh/experiment.toml
```

This creates `runs/tjh/preprocess/`, writes `manifest.json`, the split contract, and materializes binned feature views.

## 2. Train

```bash
uv run oneehr train --config examples/tjh/experiment.toml
```

The example config trains 6 models (xgboost, catboost, gru, lstm, tcn, transformer) and writes checkpoints under `runs/tjh/train/`.

## 3. Test

```bash
uv run oneehr test --config examples/tjh/experiment.toml
```

Evaluates all trained models on the held-out test split. Writes `runs/tjh/test/predictions.parquet` and `metrics.json`.

## 4. Analyze

```bash
uv run oneehr analyze --config examples/tjh/experiment.toml
```

Writes structured analysis outputs under `runs/tjh/analyze/`, including cross-system comparison and feature importance.

## Optional: Web UI

Build the frontend once:

```bash
cd webui
npm install
npm run build
```

Then serve the API and built dashboard:

```bash
cd ..
uv pip install -e ".[webui]"
uv run oneehr webui serve --root runs
```

Open `http://127.0.0.1:8000`.
