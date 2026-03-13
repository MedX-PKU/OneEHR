# OneEHR

OneEHR is a Python toolkit for longitudinal EHR predictive modeling, structured analysis, and agent-ready case workflows. It starts from standardized CSV tables and writes a stable run artifact contract that the CLI, notebooks, and Web UI all consume.

## Workflow Map

The main pipeline is:

1. `preprocess` to materialize binned and tabular views from raw event tables
2. `train` and `test` to fit and evaluate models
3. `analyze` to write structured analysis modules under `analysis/`
4. `cases build` to materialize durable case bundles
5. `query ...` and `webui serve` to read the same artifacts as JSON or a browser UI

Optional agent workflows build on the same run directory:

- `agent predict` runs OpenAI-compatible prediction backends over materialized instances
- `agent review` evaluates model or agent predictions against durable case evidence

## Start Here

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Data Model](getting-started/data-model.md)

## Choose A Workflow

- Use [Quickstart](getting-started/quickstart.md) if you want one end-to-end runnable path with the bundled example config.
- Use [Core Workflows](guide/core-workflows.md) if you want the standard preprocess/train/test/analyze/cases/query operating model.
- Use [Agent Workflows](guide/agent-workflows.md) if you are configuring OpenAI-compatible prediction or review backends.
- Use [Web UI](guide/webui.md) if you want a browser interface over analyzed runs and case artifacts.
- Use [CLI Reference](reference/cli.md) if you need the command tree and flag surface.
- Use [Configuration Reference](reference/configuration.md) if you are authoring or reviewing TOML configs.
- Use [Artifacts Reference](reference/artifacts.md) if you need the on-disk contract for notebooks, automation, or UI work.
- Use [Models Reference](reference/models.md) if you are selecting or tuning supported models.

## Operating Principles

- Event-table first: OneEHR expects a normalized long-form `dynamic.csv` rather than a dataset-specific ingestion layer.
- Leakage prevention by default: all supported split strategies are patient-level group splits.
- TOML-first configuration: the config is the experiment contract; the CLI mostly selects paths and workflow entrypoints.
- Structured outputs: public artifacts are JSON, CSV, parquet, and JSONL rather than terminal-only views.
