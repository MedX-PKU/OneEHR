# Recipes

## Train and Analyze a Standard Run

```bash
uv run oneehr preprocess --config experiment.toml
uv run oneehr train --config experiment.toml
uv run oneehr analyze --config experiment.toml
```

Use this when you want standard predictive modeling and structured post-hoc analysis.

## Build Durable Case Bundles

```bash
uv run oneehr cases build --config experiment.toml
uv run oneehr query cases list --run-dir logs/example
uv run oneehr query cases evidence --run-dir logs/example --case-id fold0:p0001
```

This is the recommended path when you want downstream chart review, notebooks, or application-layer orchestration.

## Run Agent Prediction

Add agent prediction config:

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

Run it:

```bash
uv run oneehr agent predict --config experiment.toml
uv run oneehr query agent predict-summary --run-dir logs/example
uv run oneehr query agent predict-records --run-dir logs/example --actor triage --parsed-ok true --limit 10
```

Outputs are written under `agent/predict/`.

## Run Agent Review

Add case and review config:

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

Run it:

```bash
uv run oneehr cases build --config experiment.toml
uv run oneehr agent review --config experiment.toml
uv run oneehr query agent review-summary --run-dir logs/example
uv run oneehr query agent review-records --run-dir logs/example --actor reviewer --parsed-ok true --limit 10
```

Outputs are written under `agent/review/`.

## Query Analysis Tables

```bash
uv run oneehr query analysis modules --run-dir logs/example
uv run oneehr query analysis summary --run-dir logs/example --module prediction_audit
uv run oneehr query analysis table --run-dir logs/example --module cohort_analysis --table split_roles
uv run oneehr query cohorts compare --run-dir logs/example --split fold0 --left-role train --right-role test
```

## Render Case Prompts Without Running a Backend

```bash
uv run oneehr query cases render-prompt \
  --config experiment.toml \
  --run-dir logs/example \
  --case-id fold0:p0001 \
  --template summary_v1
```

Review prompt rendering works the same way, but you must also specify the target prediction:

```bash
uv run oneehr query cases render-prompt \
  --config experiment.toml \
  --run-dir logs/example \
  --case-id fold0:p0001 \
  --template evidence_review_v1 \
  --origin model \
  --predictor-name xgboost
```
