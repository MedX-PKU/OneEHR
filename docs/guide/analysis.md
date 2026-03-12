# Analysis

`oneehr analyze` writes structured outputs under `analysis/`.

## Modules

Available modules:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `temporal_analysis`
- `interpretability`
- `agent_audit`

`agent_audit` summarizes `agent/predict/summary.json` and related artifacts. It does not analyze `agent/review/` outputs.

## Run Analysis

Run all default modules:

```bash
uv run oneehr analyze --config experiment.toml
```

Run selected modules only:

```bash
uv run oneehr analyze \
  --config experiment.toml \
  --module prediction_audit \
  --module cohort_analysis
```

Compare two runs:

```bash
uv run oneehr analyze \
  --config experiment.toml \
  --module prediction_audit \
  --compare-run logs/other_run
```

## Outputs

Each module writes:

- `analysis/<module>/summary.json`
- `analysis/<module>/*.csv` for tables
- `analysis/<module>/cases/*.parquet` for case slices, when applicable
- `analysis/<module>/plots/*.json` for plot specifications, when enabled

There is no markdown or HTML report layer in the current contract. Structured artifacts are the public interface for notebooks, automation, and future UI layers.

## Reading Analysis Results

```bash
uv run oneehr query analysis modules --run-dir logs/example
uv run oneehr query analysis summary --run-dir logs/example --module prediction_audit
uv run oneehr query analysis table --run-dir logs/example --module prediction_audit --table slices
uv run oneehr query analysis plot --run-dir logs/example --module prediction_audit --plot model_primary_metric
```

Failure-case helpers:

```bash
uv run oneehr query analysis failures --run-dir logs/example
uv run oneehr query analysis failure-cases --run-dir logs/example --module prediction_audit
uv run oneehr query analysis patient-case --run-dir logs/example --patient-id p0001
```
