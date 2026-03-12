# CLI Reference

OneEHR provides a single `oneehr` CLI with nine subcommands that map to the experiment pipeline: **preprocess**, **train**, **test**, **analyze**, **workspace**, **inspect**, **llm-preprocess**, **llm-predict**, **llm-review**.

Run `uv run oneehr --help` for the authoritative list of flags.

---

## `oneehr preprocess`

Validate input data, bin irregular events into fixed time bins, build features, and materialize tabular views.

```
oneehr preprocess --config <toml> [--overview] [--overview-top-k-codes N]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--overview` | flag | `false` | Print a dataset overview JSON to stdout |
| `--overview-top-k-codes` | `int` | `20` | Number of top codes to show in overview |

**What it does:**

1. Loads config, reads dynamic/static/label CSV files
2. Bins events into fixed time bins (`preprocess.bin_size`)
3. Selects code vocabulary per `preprocess.code_selection`
4. Builds `num__*` and `cat__*` feature columns
5. Materializes patient or time tabular views
6. Runs `labels.fn` (if configured) and joins labels onto views
7. Writes `run_manifest.json` and all preprocess artifacts

**Example:**

```bash
uv run oneehr preprocess --config examples/experiment.toml
uv run oneehr preprocess --config examples/experiment.toml --overview
```

---

## `oneehr train`

Train one or more models with optional config-driven grid search (HPO) and evaluation.

```
oneehr train --config <toml> [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--force` | flag | `false` | Overwrite existing run directory |

**What it does:**

1. Loads config and precomputed artifacts from the run directory
2. Creates patient-level group splits
3. For each model in `[[models]]`:
    - Resolves per-model HPO config (`[hpo]` or `[hpo_models.<name>]`)
    - Runs grid search if enabled
    - For each split: trains, evaluates, optionally calibrates
    - Saves model checkpoints, metrics, and predictions
4. Writes `summary.json` and `paper_table.csv`

**Example:**

```bash
uv run oneehr train --config examples/experiment.toml
uv run oneehr train --config examples/experiment.toml --force
```

---

## `oneehr test`

Evaluate trained models on a test dataset.

```
oneehr test --config <toml> [--run-dir DIR] [--test-dataset PATH] [--force] [--out-dir DIR]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--test-dataset` | `str` | `None` | Path to test dataset config (overrides `datasets.test`) |
| `--force` | flag | `false` | Overwrite existing test results |
| `--out-dir` | `str` | `None` | Output directory for test results |

**What it does:**

1. Reads the run manifest from the training run
2. Loads the test dataset and re-bins using the train-time code vocabulary
3. Aligns feature columns to match the training schema
4. For each trained model/split: loads model, generates predictions, computes metrics
5. Writes per-model/split metrics and `test_summary.json`

**Example:**

```bash
uv run oneehr test --config examples/experiment.toml
uv run oneehr test --config examples/experiment.toml --run-dir logs/example --test-dataset path/to/test.toml
```

---

## `oneehr analyze`

Run modular analysis and static reporting on an existing run directory.

```
oneehr analyze --config <toml> [--run-dir DIR] [--module NAME] [--format FMT] [--compare-run DIR] [--case-limit N] [--method xgboost|shap|attention]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--module` | repeatable `str` | from `[analysis].default_modules` | Analysis module to run |
| `--format` | repeatable `str` | from `[analysis].formats` | Extra report formats: `json`, `csv`, `md`, `html` |
| `--compare-run` | `str` | `None` | Optional second run directory for comparison reporting |
| `--case-limit` | `int` | from `[analysis].case_limit` | Max case-level rows saved per slice |
| `--method` | `str` | `None` | Compatibility shortcut for the `interpretability` module: `xgboost`, `shap`, or `attention` |

**What it does:**

1. Loads the run manifest and existing train/LLM artifacts from the run directory
2. Runs one or more analysis modules:
    - `dataset_profile`
    - `cohort_analysis`
    - `prediction_audit`
    - `temporal_analysis`
    - `interpretability`
    - `llm_audit`
3. Writes `analysis/index.json` plus per-module summaries, CSV tables, plot specs, case tables, and optional Markdown/HTML reports
4. Optionally writes `analysis/comparison/` when `--compare-run` is provided
5. Preserves legacy `analysis/feature_importance_{model}_{split}_{method}.json` outputs for interpretability compatibility

!!! note "Interpretability compatibility"
    - `--method` runs only the `interpretability` module
    - `xgboost` works with XGBoost tabular models
    - `shap` is attempted for supported tabular models; if the local SHAP stack is unavailable, the module records the error and continues
    - `attention` is reserved for models that expose attention weights and may be skipped if unavailable in the current run

**Example:**

```bash
uv run oneehr analyze --config examples/experiment.toml
uv run oneehr analyze --config examples/experiment.toml --module prediction_audit --module interpretability
uv run oneehr analyze --config examples/experiment.toml --compare-run logs/baseline_run
uv run oneehr analyze --config examples/experiment.toml --method xgboost
```

---

## `oneehr workspace`

Materialize durable, evidence-grounded case bundles for agent workflows and reviewer loops.

```
oneehr workspace --config <toml> [--run-dir DIR] [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--force` | flag | `false` | Overwrite existing workspace artifacts |

**What it does:**

1. Resolves a run directory and reuses the saved split contract
2. Builds one workspace case per held-out patient or time-window case
3. Collects evidence-grounded events, static features, predictions, and analysis refs
4. Writes `workspace/index.json` and case bundles under `workspace/cases/`

**Example:**

```bash
uv run oneehr workspace --config examples/experiment.toml
uv run oneehr workspace --config examples/experiment.toml --run-dir logs/example --force
```

---

## `oneehr inspect`

Read existing prompt, run, workspace, analysis, and review artifacts as machine-readable JSON. This command is intended for agents, scripts, notebooks, or external orchestration layers that need a stable read-only contract over OneEHR outputs.

```
oneehr inspect --tool TOOL [--config <toml> | --run-dir DIR | --root DIR] [--module NAME] [--table NAME] [--plot NAME] [--patient-id ID] [--case-id ID] [--template NAME]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tool` | `str` | *required* | Inspect contract name |
| `--config` | `str` | `None` | Optional config used to resolve the run directory |
| `--run-dir` | `str` | `None` | Run directory for run-scoped tools |
| `--root` | `str` | `"logs"` | Run root for `runs.list` |
| `--module` | `str` | `None` | Analysis/case module name |
| `--table` | `str` | `None` | Table name for `analysis.read_table` |
| `--plot` | `str` | `None` | Plot name for `analysis.read_plot_spec` |
| `--patient-id` | `str` | `None` | Patient identifier for `cases.describe_patient` |
| `--name` | `str` | `None` | Optional case artifact name filter |
| `--split` | `str` | `None` | Split name for `cohorts.compare` |
| `--left-role` | `str` | `"train"` | Left cohort role for `cohorts.compare` |
| `--right-role` | `str` | `"test"` | Right cohort role for `cohorts.compare` |
| `--limit` | `int` | `None` | Max rows returned for table/case queries |
| `--top-k` | `int` | `10` | Max feature-drift rows returned for `cohorts.compare` |
| `--template` | `str` | `None` | Prompt template name for prompt/task tools |
| `--family` | `str` | `None` | Optional prompt family filter for `prompts.list` |
| `--case-id` | `str` | `None` | Workspace case identifier for `workspace.*` and `tasks.*` |
| `--model-name` | `str` | `None` | Optional prediction model filter for task tools |
| `--source` | `str` | `None` | Optional prediction source filter (`train` or `llm`) |

Supported tools:

- `prompts.list`
- `prompts.describe`
- `runs.list`
- `runs.describe`
- `reviews.read_summary`
- `workspace.read_index`
- `workspace.list_cases`
- `workspace.read_case`
- `analysis.list_modules`
- `analysis.read_index`
- `analysis.read_summary`
- `analysis.read_table`
- `analysis.read_plot_spec`
- `cases.list_failures`
- `cases.read_failures`
- `cases.describe_patient`
- `cohorts.compare`
- `tasks.get_patient_timeline`
- `tasks.get_patient_static`
- `tasks.get_case_predictions`
- `tasks.collect_evidence`
- `tasks.render_prompt`

**What it does:**

1. Resolves a run directory from `--run-dir` or `--config` when needed
2. Loads the requested prompt/run/workspace/task/analysis/case/cohort/review artifact
3. Prints a stable JSON payload to stdout
4. Never mutates repo-tracked state

**Example:**

```bash
uv run oneehr inspect --tool prompts.list
uv run oneehr inspect --tool runs.list --root logs
uv run oneehr inspect --tool runs.describe --run-dir logs/example
uv run oneehr inspect --tool workspace.list_cases --run-dir logs/example
uv run oneehr inspect --tool tasks.collect_evidence --run-dir logs/example --case-id fold0:p0001
uv run oneehr inspect --tool analysis.read_table --run-dir logs/example --module prediction_audit --table slices --limit 5
uv run oneehr inspect --tool cases.describe_patient --run-dir logs/example --patient-id p0001
uv run oneehr inspect --tool cohorts.compare --run-dir logs/example --split fold0 --left-role train --right-role test
```

---

## `oneehr llm-preprocess`

Materialize LLM evaluation instances from the run directory. This command uses the existing patient-level split contract and writes prompt-ready instance tables under `llm/instances/`.

```
oneehr llm-preprocess --config <toml> [--run-dir DIR] [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--force` | flag | `false` | Overwrite existing LLM instance artifacts |

**What it does:**

1. Loads the existing `run_manifest.json` produced by `oneehr preprocess`
2. Reuses saved patient-group splits if present; otherwise materializes them from the current split config
3. Builds patient-level or time-level LLM instances from held-out test patients only
4. Writes `llm/instances/patient_instances.parquet` or `llm/instances/time_instances.parquet`
5. Writes `llm/instances/summary.json`

**Example:**

```bash
uv run oneehr llm-preprocess --config examples/experiment.toml
uv run oneehr llm-preprocess --config examples/experiment.toml --run-dir logs/example --force
```

---

## `oneehr llm-predict`

Render structured EHR prompts, call OpenAI-compatible chat completions, parse strict JSON outputs, and score predictions.

```
oneehr llm-predict --config <toml> [--run-dir DIR] [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--force` | flag | `false` | Overwrite existing LLM prediction artifacts |

**What it does:**

1. Loads `llm/instances/*.parquet` (materializing them first if missing)
2. Renders the selected prediction prompt template (default `summary_v1`) for each instance
3. Calls each `[[llm_models]]` entry through an OpenAI-compatible `chat/completions` endpoint
4. Parses strict JSON responses for `binary` or `regression` tasks
5. Writes prompts, raw responses, parsed outputs, predictions, failures, per-split metrics, and `llm/summary.json`

**Example:**

```bash
uv run oneehr llm-predict --config examples/experiment.toml
uv run oneehr llm-predict --config examples/experiment.toml --run-dir logs/example --force
```

---

## `oneehr llm-review`

Render reviewer prompts over materialized case workspaces, call OpenAI-compatible chat completions, parse strict review JSON outputs, and summarize judgment metrics.

```
oneehr llm-review --config <toml> [--run-dir DIR] [--force]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | *required* | Path to experiment TOML config |
| `--run-dir` | `str` | from config | Run directory (overrides `output.root/output.run_name`) |
| `--force` | flag | `false` | Overwrite existing reviewer artifacts |

**What it does:**

1. Ensures `workspace/` case bundles exist
2. Selects train and/or LLM predictions per `[review].prediction_sources`
3. Renders the selected review prompt template (default `evidence_review_v1`) for each target prediction
4. Calls each `[[review_models]]` entry through an OpenAI-compatible `chat/completions` endpoint
5. Parses structured reviewer JSON and writes prompts, raw responses, parsed outputs, grouped metrics, failures, and `review/summary.json`

**Example:**

```bash
uv run oneehr workspace --config examples/experiment.toml
uv run oneehr llm-review --config examples/experiment.toml
uv run oneehr llm-review --config examples/experiment.toml --run-dir logs/example --force
```
