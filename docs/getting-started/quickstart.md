# Quickstart

This quickstart uses the bundled example config at `examples/experiment.toml`. It walks through the standard modeling path first, then points to the optional agent and Web UI workflows.

## 1. Preprocess The Example Data

```bash
uv run oneehr preprocess --config examples/experiment.toml --overview
```

This creates `logs/example/`, writes `run_manifest.json`, and materializes the feature views used by every downstream command.

## 2. Train The Configured Models

```bash
uv run oneehr train --config examples/experiment.toml
```

The example config trains multiple models, enables HPO, and writes model artifacts, predictions, and training summaries under the same run directory.

## 3. Evaluate On The Saved Test Splits

```bash
uv run oneehr test --config examples/experiment.toml
```

By default, `oneehr test` evaluates the trained models against the run's saved split contract. The output is written under `test_runs/` inside the run directory unless you override it.

## 4. Write Structured Analysis Outputs

```bash
uv run oneehr analyze --config examples/experiment.toml
```

This writes module outputs under `analysis/`, including summaries, tables, optional plot specs, and case slices when the selected module emits them.

## 5. Materialize Durable Case Bundles

```bash
uv run oneehr cases build --config examples/experiment.toml
```

Case bundles live under `cases/` and merge available evidence, split context, and predictions into a durable review-friendly format.

## 6. Query The Run Contract

```bash
uv run oneehr query runs describe --config examples/experiment.toml
uv run oneehr query analysis summary --run-dir logs/example --module prediction_audit
uv run oneehr query cases list --run-dir logs/example
```

`query` is the structured read layer for automation, notebooks, and UI consumers. It emits JSON to stdout.

## 7. Optional Agent Workflows

After you configure `[[agent.predict.backends]]` and/or `[[agent.review.backends]]`:

```bash
uv run oneehr agent predict --config examples/experiment.toml
uv run oneehr agent review --config examples/experiment.toml
```

Inspect the resulting artifacts:

```bash
uv run oneehr query agent predict-summary --run-dir logs/example
uv run oneehr query agent review-summary --run-dir logs/example
```

## 8. Optional Web UI

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
