# Quickstart

This quickstart uses the bundled example dataset and config.

## 1. Preprocess

```bash
uv run oneehr preprocess --config examples/experiment.toml --overview
```

This creates the run directory, writes `run_manifest.json`, and materializes binned/tabular artifacts.

## 2. Train

```bash
uv run oneehr train --config examples/experiment.toml
```

Model predictions are written under `preds/` and summarized in `summary.json`.

## 3. Analyze

```bash
uv run oneehr analyze --config examples/experiment.toml
```

Analysis writes structured outputs under `analysis/`, including `summary.json`, CSV tables, parquet case slices, and optional plot specs.

## 4. Build Cases

```bash
uv run oneehr cases build --config examples/experiment.toml
```

This creates durable case bundles under `cases/`.

## 5. Query Structured Outputs

```bash
uv run oneehr query runs describe --config examples/experiment.toml
uv run oneehr query analysis summary --run-dir logs/example --module prediction_audit
uv run oneehr query cases list --run-dir logs/example
```

## 6. Optional Agent Workflows

After configuring `[[agent.predict.backends]]` and/or `[[agent.review.backends]]`:

```bash
uv run oneehr agent predict --config examples/experiment.toml
uv run oneehr agent review --config examples/experiment.toml
```

You can then query their summaries:

```bash
uv run oneehr query agent predict-summary --run-dir logs/example
uv run oneehr query agent review-summary --run-dir logs/example
```
