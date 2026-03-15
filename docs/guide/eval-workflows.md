# Evaluation Workflows

OneEHR's public evaluation surface provides unified, cross-system evaluation. It freezes a comparable EHR sample set, executes configured systems on the same saved evidence, and writes leaderboard, split, trace, and paired-comparison artifacts that are reproducible across reruns.

Use this guide when you want to compare:

- conventional ML/DL baselines already produced by `oneehr train`
- single-LLM systems
- multi-agent medical AI systems

The currently supported framework types are:

- `single_llm`
- `healthcareagent`
- `reconcile`
- `mac`
- `medagent`
- `colacare`
- `mdagents`

## Before You Start

The evaluation workflow builds on a prepared run directory:

- `eval build` requires a run produced by `oneehr preprocess`
- trained-model systems in `[[eval.systems]]` require model outputs from `oneehr train`
- framework systems require one or more configured `[[eval.backends]]`
- `analyze` is optional, but `eval.include_analysis_context = true` can only enrich evidence after analysis artifacts exist

Typical sequence:

```bash
uv run oneehr preprocess --config experiment.toml
uv run oneehr train --config experiment.toml
uv run oneehr test --config experiment.toml
uv run oneehr analyze --config experiment.toml
uv run oneehr eval build --config experiment.toml
uv run oneehr eval run --config experiment.toml
uv run oneehr eval report --config experiment.toml
```

## Minimal Config Shape

Start with a frozen-instance block plus at least one system:

```toml
[eval]
instance_unit = "patient"
max_instances = 500
include_static = true
include_analysis_context = false
max_events = 200
time_order = "asc"
primary_metric = "auroc"
bootstrap_samples = 200
save_evidence = true
save_traces = true

[[eval.systems]]
name = "xgboost_ref"
kind = "trained_model"
sample_unit = "patient"
source_model = "xgboost"
```

That is enough to benchmark a trained model baseline on the frozen eval set.

To add LLM or agent systems, define reusable backends and point systems at them:

```toml
[[eval.backends]]
name = "gpt4o-mini"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
supports_json_schema = true

[[eval.systems]]
name = "single_llm_eval"
kind = "framework"
framework_type = "single_llm"
sample_unit = "patient"
backend_refs = ["gpt4o-mini"]
max_retries = 2
timeout_seconds = 60.0

[[eval.systems]]
name = "mdagents_eval"
kind = "framework"
framework_type = "mdagents"
sample_unit = "patient"
backend_refs = ["gpt4o-mini"]
max_rounds = 2
framework_params = { num_teams_advanced = 2, num_agents_per_team_advanced = 2 }

[[eval.suites]]
name = "core"
primary_metric = "auroc"
compare_pairs = [["xgboost_ref", "single_llm_eval"], ["xgboost_ref", "mdagents_eval"]]
```

Rules that matter:

- `eval.instance_unit` must be `patient` or `time`
- every `eval.systems.sample_unit` must match `eval.instance_unit`
- `kind = "trained_model"` systems require `source_model`
- `kind = "framework"` systems require `framework_type` plus valid `backend_refs`

## Step 1: Freeze The Eval Set

`oneehr eval build` creates the shared comparison set.

```bash
uv run oneehr eval build --config experiment.toml
uv run oneehr eval build --config experiment.toml --run-dir logs/example --force
```

It writes:

- `eval/index.json`
- `eval/instances/instances.parquet`
- `eval/evidence/...` when `save_evidence = true`

This is the fairness boundary of the system. Every compared system should see the same frozen records and the same saved evidence bundle.

## Step 2: Run The Systems

`oneehr eval run` executes every enabled system in `[[eval.systems]]`.

```bash
uv run oneehr eval run --config experiment.toml
uv run oneehr eval run --config experiment.toml --run-dir logs/example --force
```

Outputs are written under:

- `eval/predictions/{system_name}/predictions.parquet`
- `eval/traces/{system_name}/trace.parquet` when traces are saved
- `eval/summary.json`

Trained-model systems reuse saved model predictions. Framework systems call the configured OpenAI-compatible backends and store both predictions and structured trace rows.

## Step 3: Build The Report

`oneehr eval report` computes the comparison artifacts.

```bash
uv run oneehr eval report --config experiment.toml
uv run oneehr eval report --config experiment.toml --run-dir logs/example --force
```

It writes:

- `eval/reports/leaderboard.csv`
- `eval/reports/split_metrics.csv`
- `eval/reports/pairwise.csv`
- `eval/reports/summary.json`

Use `eval.suites.compare_pairs` when you want explicit paired deltas between named systems. If no suites are configured, OneEHR still writes the default report tables.

## Query And Inspection

Use `query eval` for JSON access:

```bash
uv run oneehr query eval index --run-dir logs/example
uv run oneehr query eval summary --run-dir logs/example
uv run oneehr query eval report --run-dir logs/example
uv run oneehr query eval table --run-dir logs/example --table leaderboard --limit 10
uv run oneehr query eval instance --run-dir logs/example --instance-id fold0:p0001
uv run oneehr query eval trace --run-dir logs/example --system mdagents_eval --stage complexity_routing
```

Or use the shorter CLI aliases for two common drill-downs:

```bash
uv run oneehr eval instance --config experiment.toml --instance-id fold0:p0001
uv run oneehr eval trace --config experiment.toml --system mdagents_eval --stage complexity_routing
```

Built-in prompt templates remain inspectable through:

```bash
uv run oneehr query prompts list
uv run oneehr query prompts describe --template summary_v1
```

## Reproducibility And Fairness

The unified evaluation design is intended to make cross-system claims defensible:

- one frozen `instances.parquet` anchors the sample set
- one saved evidence bundle per instance anchors what each framework saw
- one explicit `[[eval.backends]]` block records backend model, endpoint, and optional cost settings
- one `config_sha256` is persisted into predictions and traces for system-level provenance
- one scoring pass produces the leaderboard, split tables, and pairwise deltas

That does not remove all sources of variance, but it makes them visible and replayable.
