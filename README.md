# OneEHR

OneEHR is a Python toolkit for longitudinal EHR modeling, structured analysis, and reproducible evaluation across trained models, single-LLM systems, and multi-agent medical frameworks. It starts from standardized CSV tables and writes one run contract that can be consumed from the CLI, notebooks, and the first-party web/API layer.

## Workflow At A Glance

OneEHR is organized around one artifact contract:

- `preprocess` materializes the binned and tabular views used by every downstream workflow and saves the split contract under `splits/`.
- `train` and `test` run classical ML or DL modeling from a TOML experiment config.
- `analyze` writes structured module outputs under `analysis/`.
- `eval build`, `eval run`, and `eval report` freeze evaluation instances, execute configured systems on the same evidence, and write reproducible comparison outputs under `eval/`.
- `query ...` and `webui serve` expose the same artifacts as JSON or a browser-backed API.

The top-level CLI surface is:

- `oneehr preprocess`
- `oneehr train`
- `oneehr test`
- `oneehr analyze`
- `oneehr eval ...`
- `oneehr query ...`
- `oneehr webui serve`

Inspect the live interface with:

```bash
uv run oneehr --help
uv run oneehr eval --help
uv run oneehr query --help
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
uv run oneehr eval build --config examples/experiment.toml
uv run oneehr eval run --config examples/experiment.toml
uv run oneehr eval report --config examples/experiment.toml
uv run oneehr query runs describe --config examples/experiment.toml
uv run oneehr query eval report --run-dir logs/example
```

This writes the run under `logs/example/`, including `run_manifest.json`, `splits/`, model outputs, `analysis/`, and `eval/`.

The bundled example config ships with a trained-model evaluation baseline. To compare LLM or multi-agent systems on the same frozen instances, add `[[eval.backends]]` plus additional `[[eval.systems]]` entries and the corresponding API key environment variables.

Supported framework types in the current eval surface:

- `single_llm`
- `healthcareagent`
- `reconcile`
- `mac`
- `medagent`
- `colacare`
- `mdagents`

Optional web/API workflow:

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
- `[analysis]` for structured reporting modules
- `[eval]`, `[[eval.backends]]`, `[[eval.systems]]`, and `[[eval.suites]]` for unified evaluation
- `[output]` for run root and run name

Legacy `[cases]`, `[agent.predict]`, and `[agent.review]` sections are no longer supported.

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
- [`docs/guide/eval-workflows.md`](docs/guide/eval-workflows.md)
- [`docs/guide/webui.md`](docs/guide/webui.md)
- [`docs/reference/cli.md`](docs/reference/cli.md)
- [`docs/reference/configuration.md`](docs/reference/configuration.md)
- [`docs/reference/artifacts.md`](docs/reference/artifacts.md)

The canonical step-by-step walkthrough lives in [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md).

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
uv run oneehr eval --help
uv run oneehr preprocess --config examples/experiment.toml --overview
uv run mkdocs build
```
