# OneEHR

OneEHR is a Python toolkit for longitudinal EHR predictive modeling, analysis, and agent-ready case workflows.

The project is organized around three layers:

- `preprocess` / `train` / `test` / `analyze` for classical ML and DL modeling.
- `cases build`, `agent predict`, and `agent review` for evidence-grounded case bundles and agent workflows.
- `query ...` and `webui serve` for stable JSON/JSONL-oriented access and a first-party analysis dashboard.

## Why OneEHR

- Event-table first: start from a long-form EHR table with `patient_id`, `event_time`, `code`, and `value`.
- Leakage prevention by default: splits are always patient-level group splits.
- TOML-first configuration: the config is the source of truth; CLI flags are only for paths and overrides.
- Structured artifacts: outputs are written as JSON, CSV, parquet, and JSONL so they can be analyzed programmatically.

## Installation

OneEHR requires Python 3.12 and `uv`.

```bash
uv venv .venv --python 3.12
uv pip install -e .
```

Optional docs dependencies:

```bash
uv pip install -e ".[docs]"
```

Optional Web UI backend dependencies:

```bash
uv pip install -e ".[webui]"
```

## Quickstart

Use the bundled example dataset and config:

```bash
uv run oneehr preprocess --config examples/experiment.toml --overview
uv run oneehr train --config examples/experiment.toml
uv run oneehr analyze --config examples/experiment.toml
uv run oneehr cases build --config examples/experiment.toml
uv run oneehr query runs describe --config examples/experiment.toml
```

Optional agent workflows:

```bash
uv run oneehr agent predict --config examples/experiment.toml
uv run oneehr agent review --config examples/experiment.toml
```

`agent predict` and `agent review` require configured OpenAI-compatible backends and the corresponding API key environment variables.

Optional Web UI workflow:

```bash
cd webui
npm install
npm run build
cd ..
uv run oneehr webui serve --root logs
```

Then open `http://127.0.0.1:8000`.

## Command Surface

Top-level commands:

- `oneehr preprocess`
- `oneehr train`
- `oneehr test`
- `oneehr analyze`
- `oneehr cases build`
- `oneehr agent predict`
- `oneehr agent review`
- `oneehr query ...`
- `oneehr webui serve`

Explore the live CLI:

```bash
uv run oneehr --help
uv run oneehr query --help
uv run oneehr query cases --help
uv run oneehr webui serve --help
```

## Web UI

The Web UI is a `React + FastAPI` workspace console for analyzed runs. It reads only from the existing run artifact contract through `/api/v1`, so the browser never reads files directly from disk.

Backend:

```bash
uv pip install -e ".[webui]"
uv run oneehr webui serve --root logs
```

Frontend build:

```bash
cd webui
npm install
npm run build
```

Frontend dev server:

```bash
cd webui
npm run dev
```

By default the Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`.

Current workspace surfaces:

- run overview and analysis module dashboards
- durable case bundle browsing and case detail views
- agent predict/review summaries
- compare-run views when `analysis/comparison/*` exists

## Data Model

Primary dynamic table:

| column | meaning |
| --- | --- |
| `patient_id` | patient identifier |
| `event_time` | event timestamp |
| `code` | event code |
| `value` | numeric or categorical event value |

Optional static table:

| column | meaning |
| --- | --- |
| `patient_id` | patient identifier |
| other columns | patient-level covariates |

Labels can come from:

- a standard label table
- a user label function via `[labels].fn`

Prediction modes:

- `patient`: N-1 / patient-level prediction
- `time`: N-N / time-window prediction

## Configuration Overview

Important sections:

- `[dataset]` or `[datasets]`: input tables
- `[preprocess]`: binning and feature construction
- `[task]`: `binary` or `regression`, `patient` or `time`
- `[labels]`: label function configuration
- `[split]`: patient-level split strategy
- `[model]` or `[[models]]`: training models for `oneehr train`
- `[trainer]`: optimization and training controls
- `[analysis]`: default modules and analysis settings
- `[cases]`: durable case bundle materialization
- `[agent.predict]`: agent prediction workflow
- `[agent.review]`: agent review workflow
- `[output]`: run directory root and run name

Minimal training config:

```toml
[dataset]
dynamic = "examples/dynamic.csv"
static = "examples/static.csv"

[preprocess]
bin_size = "1d"
numeric_strategy = "mean"
categorical_strategy = "onehot"
code_selection = "frequency"
top_k_codes = 100

[task]
kind = "binary"
prediction_mode = "patient"

[labels]
fn = "examples/label_fn.py:build_labels"

[split]
kind = "kfold"
n_splits = 5
seed = 42

[model]
name = "xgboost"

[trainer]
device = "cpu"

[output]
root = "logs"
run_name = "example"
```

Minimal agent prediction add-on:

```toml
[agent.predict]
enabled = true
sample_unit = "patient"
prompt_template = "summary_v1"

[agent.predict.prompt]
include_static = true
max_events = 50

[agent.predict.output]
include_explanation = true

[[agent.predict.backends]]
name = "gpt4o-mini"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

Minimal agent review add-on:

```toml
[cases]
include_static = true
include_analysis_refs = true

[agent.review]
enabled = true
prompt_template = "evidence_review_v1"
prediction_origins = ["model", "agent"]

[agent.review.prompt]
include_static = true
include_ground_truth = true
include_analysis_context = true

[[agent.review.backends]]
name = "reviewer"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

## Artifact Layout

Each run lives under `[output].root / [output].run_name`.

Key outputs:

```text
logs/<run_name>/
├── run_manifest.json
├── summary.json
├── analysis/
├── cases/
├── agent/
│   ├── predict/
│   └── review/
├── preds/
├── models/
└── splits/
```

Important structured artifacts:

- `run_manifest.json`: run contract snapshot
- `summary.json`: training summary
- `analysis/<module>/summary.json`: analysis module summary
- `analysis/<module>/*.csv`: analysis tables
- `cases/index.json`: run-level case index
- `cases/<case_slug>/case.json`: case metadata
- `cases/<case_slug>/events.csv`: event evidence
- `cases/<case_slug>/predictions.csv`: merged model and agent predictions
- `agent/predict/summary.json`: agent prediction summary
- `agent/review/summary.json`: agent review summary

The `query` command reads these artifacts directly rather than rendering a terminal-specific view.

## Query Workflows

List runs:

```bash
uv run oneehr query runs list --root logs
```

Describe a run:

```bash
uv run oneehr query runs describe --run-dir logs/example
```

Read case bundles:

```bash
uv run oneehr query cases list --run-dir logs/example
uv run oneehr query cases read --run-dir logs/example --case-id fold0:p0001
uv run oneehr query cases evidence --run-dir logs/example --case-id fold0:p0001
```

Read analysis artifacts:

```bash
uv run oneehr query analysis modules --run-dir logs/example
uv run oneehr query analysis summary --run-dir logs/example --module prediction_audit
uv run oneehr query analysis table --run-dir logs/example --module prediction_audit --table slices
```

Read agent summaries:

```bash
uv run oneehr query agent predict-summary --run-dir logs/example
uv run oneehr query agent review-summary --run-dir logs/example
```

## Documentation

Start the docs locally:

```bash
uv run mkdocs serve
```

Build the docs:

```bash
uv run mkdocs build
```

## Validation

Recommended checks:

```bash
uv run oneehr --help
uv run pytest -q
uv run oneehr preprocess --config examples/experiment.toml --overview
uv run mkdocs build
```
