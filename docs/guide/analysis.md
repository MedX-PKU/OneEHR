# Analysis

`oneehr analyze` reads an existing run directory and writes a modular analysis bundle under `analysis/`.

The default suite combines:

- `dataset_profile`
- `cohort_analysis`
- `prediction_audit`
- `temporal_analysis`
- `interpretability`
- `llm_audit`

---

## Basic usage

```bash
uv run oneehr analyze --config experiment.toml
```

This writes:

- `analysis/index.json`
- `analysis/<module>/summary.json`
- `analysis/<module>/*.csv` when `csv` output is enabled
- `analysis/<module>/plots/*.json` when plot specs are enabled
- `analysis/<module>/cases/*.parquet` for case-level audit exports
- `analysis/<module>/summary.md` and `summary.html` when those formats are enabled

---

## Selecting modules

Run only specific modules:

```bash
uv run oneehr analyze --config experiment.toml --module prediction_audit --module interpretability
```

Useful modules:

- `dataset_profile`: raw event counts, feature-space summary, top codes, static missingness
- `cohort_analysis`: split-role composition and feature drift against the training cohort
- `prediction_audit`: per-model/per-split error summaries, subgroup metrics, and case tables
- `temporal_analysis`: performance sliced by event-count quantiles and, for time-mode runs, by `bin_time`
- `interpretability`: tabular feature-importance exports plus legacy compatibility files
- `llm_audit`: parse success, coverage, token usage, latency, and failure buckets for LLM runs

---

## Report formats

Override the configured formats from the CLI:

```bash
uv run oneehr analyze --config experiment.toml --format json --format csv
```

Notes:

- `analysis/index.json` and each module `summary.json` are always written.
- `csv` enables table exports.
- `md` and `html` enable static human-readable reports.

---

## Compare runs

Generate the requested modules for the current run and also write a metric delta report against another run:

```bash
uv run oneehr analyze \
  --config experiment.toml \
  --module prediction_audit \
  --compare-run logs/baseline_run
```

Comparison outputs are written under `analysis/comparison/`.

Current implementation compares:

- train-summary metrics from `summary.json`
- LLM summary metrics from `llm/summary.json` when both runs have LLM outputs

The compared runs must have matching task settings.

---

## Interpretability compatibility

The previous `analyze` behavior is preserved as a compatibility path:

```bash
uv run oneehr analyze --config experiment.toml --method xgboost
uv run oneehr analyze --config experiment.toml --method shap
```

Behavior:

- `--method` runs only the `interpretability` module
- legacy files are still written as `analysis/feature_importance_{model}_{split}_{method}.json`
- the same results are also indexed under `analysis/interpretability/`

Notes:

- `xgboost` works with XGBoost tabular models
- `shap` is attempted for supported tabular models and may be skipped if the local SHAP stack is unavailable
- `attention` is reserved for models that expose attention weights and may be skipped if the run does not provide them

---

## Configuring defaults

Use `[analysis]` in TOML to control the default module set and outputs:

```toml
[analysis]
default_modules = [
  "dataset_profile",
  "cohort_analysis",
  "prediction_audit",
  "temporal_analysis",
  "interpretability",
  "llm_audit",
]
formats = ["json", "csv", "md", "html"]
top_k = 20
stratify_by = []
case_limit = 50
save_plot_specs = true
shap_max_samples = 500
```

`stratify_by` expects column names from `static.csv`; those columns are joined onto prediction rows for subgroup metrics in `prediction_audit`.
