# OneEHR

OneEHR is a toolkit for longitudinal EHR modeling, analysis, and agent-ready case workflows.

Core workflow:

- `preprocess` builds binned and tabular views from event tables.
- `train` fits ML or DL models.
- `test` evaluates trained models on held-out or external data.
- `analyze` writes structured analysis outputs under `analysis/`.
- `cases build` materializes durable evidence bundles under `cases/`.
- `agent predict` and `agent review` add OpenAI-compatible agent workflows.
- `query ...` reads run artifacts as JSON for notebooks, automation, and future web UIs.

Start here:

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [CLI Reference](reference/cli.md)
- [Configuration Reference](reference/configuration.md)
- [Artifacts Reference](reference/artifacts.md)
