# CLI Reference

OneEHR exposes four top-level commands:

- `preprocess`
- `train`
- `test`
- `analyze`

View the live interface with:

```bash
uv run oneehr --help
```

## `oneehr preprocess`

```bash
uv run oneehr preprocess --config <toml>
```

Bins dynamic events, generates labels, splits patients, and writes the run manifest. This is the required first step for every run.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |

Outputs are written to `{output.root}/{output.run_name}/preprocess/`.

## `oneehr train`

```bash
uv run oneehr train --config <toml> [--force]
```

Trains all models defined in `[[models]]` and writes checkpoints under `{run_dir}/train/{model_name}/`.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--force` | No | Overwrite existing train directory |

## `oneehr test`

```bash
uv run oneehr test --config <toml> [--force]
```

Runs all trained models and configured `[[systems]]` on the held-out test split. Writes `predictions.parquet` and `metrics.json` under `{run_dir}/test/`.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--force` | No | Overwrite existing test directory |

## `oneehr analyze`

```bash
uv run oneehr analyze --config <toml> [--module NAME]
```

Reads `test/predictions.parquet` and writes structured analysis outputs under `{run_dir}/analyze/`.

| Flag | Required | Description |
|------|:---:|-------------|
| `--config` | Yes | Path to TOML experiment config |
| `--module` | No | Run a single analysis module instead of all |

Available modules:

- `comparison` -- cross-system metrics comparison
- `feature_importance` -- native importance for tree models, SHAP fallback for others
