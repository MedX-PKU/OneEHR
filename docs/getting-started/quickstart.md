# Quickstart

This quickstart uses the bundled example config at `examples/experiment.toml`. It walks through the standard modeling path first, then runs the unified evaluation workflow on the resulting run artifacts.

## 1. Preprocess The Example Data

```bash
uv run oneehr preprocess --config examples/experiment.toml --overview
```

This creates `logs/example/`, writes `run_manifest.json`, saves `splits/`, and materializes the feature views used by every downstream command.

## 2. Train The Configured Models

```bash
uv run oneehr train --config examples/experiment.toml
```

The example config trains multiple models, enables HPO, and writes model artifacts, predictions, and training summaries under the same run directory.

## 3. Evaluate On The Saved Test Splits

```bash
uv run oneehr test --config examples/experiment.toml
```

By default, `oneehr test` evaluates the configured ML/DL models against the run's saved split contract. The output is written under `test_runs/` inside the run directory unless you override it.

## 4. Write Structured Analysis Outputs

```bash
uv run oneehr analyze --config examples/experiment.toml
```

This writes module outputs under `analysis/`, including summaries, tables, optional plot specs, and case slices when the selected module emits them.

## 5. Freeze Evaluation Instances

```bash
uv run oneehr eval build --config examples/experiment.toml
```

This materializes `eval/index.json`, `eval/instances/instances.parquet`, and, by default, saved evidence bundles under `eval/evidence/`.

## 6. Execute The Configured Evaluation Systems

```bash
uv run oneehr eval run --config examples/experiment.toml
```

The bundled example config includes a trained-model baseline system, so this step works without any external API keys. To compare LLM or multi-agent systems on the same frozen instances, add `[[eval.backends]]` and more `[[eval.systems]]` entries to the config.

## 7. Build The Comparison Report

```bash
uv run oneehr eval report --config examples/experiment.toml
```

This writes `eval/reports/leaderboard.csv`, `eval/reports/split_metrics.csv`, `eval/reports/pairwise.csv`, and `eval/reports/summary.json`.

## 8. Query The Run Contract

```bash
uv run oneehr query runs describe --config examples/experiment.toml
uv run oneehr query analysis summary --run-dir logs/example --module prediction_audit
uv run oneehr query eval report --run-dir logs/example
uv run oneehr query eval table --run-dir logs/example --table leaderboard
```

`query` is the structured read layer for automation, notebooks, and UI consumers. It emits JSON to stdout.

## 9. Optional Web UI

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
uv run oneehr webui serve --root logs
```

Open `http://127.0.0.1:8000`.
