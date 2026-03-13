# CLI Reference

OneEHR exposes eight top-level command groups:

- `preprocess`
- `train`
- `test`
- `analyze`
- `cases`
- `agent`
- `query`
- `webui`

Use them as three layers:

- run-building: `preprocess`, `train`, `test`, `analyze`
- case and agent workflows: `cases`, `agent`
- read-only consumption: `query`, `webui`

View the live interface with:

```bash
uv run oneehr --help
```

## `oneehr preprocess`

```bash
uv run oneehr preprocess --config <toml> [--overview] [--overview-top-k-codes N]
```

Runs preprocessing and, optionally, prints a dataset overview JSON payload.

## `oneehr train`

```bash
uv run oneehr train --config <toml> [--force]
```

Trains configured models and writes model artifacts, predictions, and `summary.json`.

## `oneehr test`

```bash
uv run oneehr test --config <toml> [--run-dir DIR] [--test-dataset PATH] [--force] [--out-dir DIR]
```

Evaluates trained models on held-out or external test data.

## `oneehr analyze`

```bash
uv run oneehr analyze --config <toml> [--run-dir DIR] [--module NAME] [--compare-run DIR] [--case-limit N] [--method xgboost|shap|attention]
```

Writes structured analysis outputs under `analysis/`.

Supported modules:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `temporal_analysis`
- `interpretability`
- `agent_audit`

## `oneehr cases`

```bash
uv run oneehr cases build --config <toml> [--run-dir DIR] [--force]
```

Materializes durable case bundles under `cases/`.

## `oneehr agent`

Prediction:

```bash
uv run oneehr agent predict --config <toml> [--run-dir DIR] [--force]
```

Review:

```bash
uv run oneehr agent review --config <toml> [--run-dir DIR] [--force]
```

`agent predict` writes under `agent/predict/`.

`agent review` writes under `agent/review/`.

## `oneehr query`

`query` is the structured read layer over existing artifacts. It returns JSON to stdout.

### Runs

```bash
uv run oneehr query runs list [--root DIR]
uv run oneehr query runs describe [--config <toml> | --run-dir DIR]
```

### Prompts

```bash
uv run oneehr query prompts list [--family prediction|review]
uv run oneehr query prompts describe --template NAME
```

### Cases

```bash
uv run oneehr query cases index [--config <toml> | --run-dir DIR]
uv run oneehr query cases list [--config <toml> | --run-dir DIR] [--limit N]
uv run oneehr query cases read [--config <toml> | --run-dir DIR] --case-id ID [--limit N]
uv run oneehr query cases timeline [--config <toml> | --run-dir DIR] --case-id ID [--limit N]
uv run oneehr query cases static [--config <toml> | --run-dir DIR] --case-id ID
uv run oneehr query cases predictions [--config <toml> | --run-dir DIR] --case-id ID [--origin model|agent] [--predictor-name NAME] [--limit N]
uv run oneehr query cases evidence [--config <toml> | --run-dir DIR] --case-id ID [--limit N]
uv run oneehr query cases render-prompt [--config <toml> | --run-dir DIR] --case-id ID [--template NAME] [--origin model|agent] [--predictor-name NAME] [--limit N]
```

### Analysis

```bash
uv run oneehr query analysis modules [--config <toml> | --run-dir DIR]
uv run oneehr query analysis index [--config <toml> | --run-dir DIR]
uv run oneehr query analysis summary [--config <toml> | --run-dir DIR] --module NAME
uv run oneehr query analysis table [--config <toml> | --run-dir DIR] --module NAME --table NAME [--limit N]
uv run oneehr query analysis plot [--config <toml> | --run-dir DIR] --module NAME --plot NAME
uv run oneehr query analysis failures [--config <toml> | --run-dir DIR] [--module NAME]
uv run oneehr query analysis failure-cases [--config <toml> | --run-dir DIR] [--module NAME] [--name NAME] [--limit N]
uv run oneehr query analysis patient-case [--config <toml> | --run-dir DIR] --patient-id ID [--module NAME] [--limit N]
```

### Cohorts

```bash
uv run oneehr query cohorts compare [--config <toml> | --run-dir DIR] [--split NAME] [--left-role train|val|test] [--right-role train|val|test] [--top-k N]
```

### Agent Artifacts

```bash
uv run oneehr query agent predict-summary [--config <toml> | --run-dir DIR]
uv run oneehr query agent review-summary [--config <toml> | --run-dir DIR]
uv run oneehr query agent predict-records [--config <toml> | --run-dir DIR] [--actor NAME] [--split NAME] [--parsed-ok true|false] [--search TEXT] [--limit N] [--offset N]
uv run oneehr query agent review-records [--config <toml> | --run-dir DIR] [--actor NAME] [--split NAME] [--parsed-ok true|false] [--search TEXT] [--limit N] [--offset N]
uv run oneehr query agent predict-failures [--config <toml> | --run-dir DIR] [--actor NAME] [--split NAME] [--search TEXT] [--limit N] [--offset N]
uv run oneehr query agent review-failures [--config <toml> | --run-dir DIR] [--actor NAME] [--split NAME] [--search TEXT] [--limit N] [--offset N]
```

## `oneehr webui`

```bash
uv run oneehr webui serve [--root DIR] [--host HOST] [--port PORT] [--frontend-dist DIR] [--reload]
```

Serves the FastAPI backend and, when `webui/dist` exists, the built React dashboard.

Typical workflow:

```bash
uv pip install -e ".[webui]"
cd webui && npm install && npm run build
cd ..
uv run oneehr webui serve --root logs
```
