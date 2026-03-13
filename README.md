# OneEHR

OneEHR is a Python toolkit for longitudinal EHR predictive modeling, structured run analysis, and agent-ready case workflows. It starts from standardized CSV tables and writes reproducible run artifacts that can be consumed from the CLI, notebooks, and the first-party Web UI.

## Workflow At A Glance

OneEHR is organized around one artifact contract:

- `preprocess` materializes the binned and tabular views used by every downstream workflow.
- `train` and `test` run classical ML or DL modeling from a TOML experiment config.
- `analyze` writes structured module outputs under `analysis/`.
- `cases build`, `agent predict`, and `agent review` add durable case bundles and OpenAI-compatible agent workflows.
- `query ...` and `webui serve` expose the same artifacts as JSON or a browser UI.

The top-level CLI surface is:

- `oneehr preprocess`
- `oneehr train`
- `oneehr test`
- `oneehr analyze`
- `oneehr cases build`
- `oneehr agent predict`
- `oneehr agent review`
- `oneehr query ...`
- `oneehr webui serve`

Inspect the live interface with:

```bash
uv run oneehr --help
uv run oneehr query --help
uv run oneehr agent --help
```

## Install

OneEHR requires Python 3.12 and `uv`.

```bash
uv venv .venv --python 3.12
uv pip install -e .
uv run oneehr --help
```

Optional extras:

```bash
uv pip install -e ".[docs]"
uv pip install -e ".[webui]"
```

## Quickstart

Use the bundled example config at [`examples/experiment.toml`](examples/experiment.toml):

```bash
uv run oneehr preprocess --config examples/experiment.toml --overview
uv run oneehr train --config examples/experiment.toml
uv run oneehr test --config examples/experiment.toml
uv run oneehr analyze --config examples/experiment.toml
uv run oneehr cases build --config examples/experiment.toml
uv run oneehr query runs describe --config examples/experiment.toml
```

This writes the run under `logs/example/`, including `run_manifest.json`, model outputs, `analysis/`, and `cases/`.

Optional agent workflows require configured `[[agent.predict.backends]]` and/or `[[agent.review.backends]]` plus the corresponding API key environment variables:

```bash
uv run oneehr agent predict --config examples/experiment.toml
uv run oneehr agent review --config examples/experiment.toml
```

Optional Web UI workflow:

```bash
cd webui
npm install
npm run build
cd ..
uv run oneehr webui serve --root logs
```

Then open `http://127.0.0.1:8000`.

## Configuration Model

OneEHR is TOML-first. CLI flags are primarily for locating configs, run directories, and a small number of overrides. The main experiment sections are:

- `[dataset]` or `[datasets]` for standardized input tables
- `[preprocess]` for binning, vocabulary selection, and post-split feature transforms
- `[task]`, `[labels]`, and `[split]` for task definition and leakage-safe evaluation setup
- `[model]` or `[[models]]`, `[trainer]`, `[hpo]`, and `[calibration]` for training
- `[analysis]`, `[cases]`, `[agent.predict]`, and `[agent.review]` for downstream workflows
- `[output]` for run root and run name

The standard input model is:

- `dynamic.csv` required: long-form event table with `patient_id`, `event_time`, `code`, and `value`
- `static.csv` optional: patient-level covariates keyed by `patient_id`
- `label.csv` optional: label events keyed by `patient_id` and `label_time`

Prediction modes:

- `patient`: patient-level N-1 prediction
- `time`: time-window N-N prediction

## Documentation

Start with:

- [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
- [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md)
- [`docs/getting-started/data-model.md`](docs/getting-started/data-model.md)
- [`docs/guide/core-workflows.md`](docs/guide/core-workflows.md)
- [`docs/guide/agent-workflows.md`](docs/guide/agent-workflows.md)
- [`docs/guide/webui.md`](docs/guide/webui.md)
- [`docs/reference/cli.md`](docs/reference/cli.md)
- [`docs/reference/configuration.md`](docs/reference/configuration.md)
- [`docs/reference/artifacts.md`](docs/reference/artifacts.md)

Build the docs locally:

```bash
uv pip install -e ".[docs]"
uv run mkdocs serve
uv run mkdocs build
```

## Validation

Recommended checks:

```bash
uv run oneehr --help
uv run oneehr preprocess --config examples/experiment.toml --overview
uv run mkdocs build
```
