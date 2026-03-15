# CLI Reference

OneEHR exposes seven top-level command groups:

- `preprocess`
- `train`
- `test`
- `analyze`
- `eval`
- `query`
- `webui`

Use them as three layers:

- run-building: `preprocess`, `train`, `test`, `analyze`
- unified evaluation: `eval`
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

Common module names:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `test_audit`
- `temporal_analysis`
- `interpretability`

## `oneehr eval`

Build frozen instances:

```bash
uv run oneehr eval build --config <toml> [--run-dir DIR] [--force]
```

Run configured systems:

```bash
uv run oneehr eval run --config <toml> [--run-dir DIR] [--force]
```

Build leaderboard and pairwise reports:

```bash
uv run oneehr eval report --config <toml> [--run-dir DIR] [--force]
```

Inspect one system trace:

```bash
uv run oneehr eval trace --config <toml> [--run-dir DIR] --system NAME [--limit N] [--offset N] [--stage NAME] [--role NAME] [--round N]
```

Inspect one frozen instance with aligned outputs:

```bash
uv run oneehr eval instance --config <toml> [--run-dir DIR] --instance-id ID
```

## `oneehr query`

`query` is the structured read layer over existing artifacts. It returns JSON to stdout.

Run-scoped commands accept either:

- `--config <toml>` to resolve the run directory from the experiment config
- `--run-dir DIR` to point at an existing run directly

### Runs

```bash
uv run oneehr query runs list [--root DIR]
uv run oneehr query runs describe [--config <toml> | --run-dir DIR]
```

### Prompts

```bash
uv run oneehr query prompts list [--family NAME]
uv run oneehr query prompts describe --template NAME
```

### Eval

```bash
uv run oneehr query eval index [--config <toml> | --run-dir DIR]
uv run oneehr query eval summary [--config <toml> | --run-dir DIR]
uv run oneehr query eval report [--config <toml> | --run-dir DIR]
uv run oneehr query eval table [--config <toml> | --run-dir DIR] --table leaderboard|split_metrics|pairwise [--limit N] [--offset N]
uv run oneehr query eval instance [--config <toml> | --run-dir DIR] --instance-id ID
uv run oneehr query eval trace [--config <toml> | --run-dir DIR] --system NAME [--limit N] [--offset N] [--stage NAME] [--role NAME] [--round N]
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

## `oneehr webui`

```bash
uv run oneehr webui serve [--root DIR] [--host HOST] [--port PORT] [--frontend-dist DIR] [--reload]
```

Serves the FastAPI backend and, when `webui/dist` exists, the built frontend bundle.

Typical workflow:

```bash
uv pip install -e ".[webui]"
cd webui && npm install && npm run build
cd ..
uv run oneehr webui serve --root logs
```
