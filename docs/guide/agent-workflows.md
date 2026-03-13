# Agent Workflows

OneEHR's agent layer runs on top of the same artifact contract used by the modeling workflows. It supports OpenAI-compatible chat-completion backends for prediction and review without introducing a separate database or dataset format.

Use this guide for when and how to run agent workflows. Use the [Configuration Reference](../reference/configuration.md) for full parameter tables.

## Before You Start

Agent workflows build on a prepared run directory:

- `agent predict` requires a preprocessed run plus configured backends
- `agent review` requires durable case bundles plus predictions to review
- `query` and `webui` can inspect agent artifacts after they are written

Typical sequence:

```bash
uv run oneehr preprocess --config experiment.toml
uv run oneehr train --config experiment.toml
uv run oneehr analyze --config experiment.toml
uv run oneehr cases build --config experiment.toml
```

## Agent Predict

`oneehr agent predict` materializes prediction instances and sends them to one or more configured backends.

Minimal config shape:

```toml
[agent.predict]
enabled = true
sample_unit = "patient"
prompt_template = "summary_v1"

[agent.predict.prompt]
include_static = true
max_events = 100

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
uv run oneehr agent predict --config experiment.toml --run-dir logs/example --force
```

Inspect the results:

```bash
uv run oneehr query agent predict-summary --run-dir logs/example
uv run oneehr query agent predict-records --run-dir logs/example --actor gpt4o-mini --parsed-ok true --limit 10
uv run oneehr query agent predict-failures --run-dir logs/example --actor gpt4o-mini
```

Outputs are written under `agent/predict/`, including prompts, raw responses, parsed records, metrics, and failure rows.

## Agent Review

`oneehr agent review` scores model or agent predictions against durable case evidence. Cases must exist first.

Minimal config shape:

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
```

Inspect the review outputs:

```bash
uv run oneehr query agent review-summary --run-dir logs/example
uv run oneehr query agent review-records --run-dir logs/example --actor reviewer --parsed-ok true --limit 10
uv run oneehr query agent review-failures --run-dir logs/example --actor reviewer
```

Outputs are written under `agent/review/`.

## Prompt Inspection And Debugging

Inspect built-in prompt templates:

```bash
uv run oneehr query prompts list
uv run oneehr query prompts describe --template summary_v1
uv run oneehr query prompts describe --template evidence_review_v1
```

Render prompts without sending a backend request:

```bash
uv run oneehr query cases render-prompt \
  --run-dir logs/example \
  --case-id fold0:p0001 \
  --template summary_v1
```

For review prompts, specify the prediction target:

```bash
uv run oneehr query cases render-prompt \
  --run-dir logs/example \
  --case-id fold0:p0001 \
  --template evidence_review_v1 \
  --origin model \
  --predictor-name xgboost
```

## Guardrails And Constraints

- `agent.predict.sample_unit` must match `[task].prediction_mode`
- built-in prediction templates keep `include_labels_context = false` in v1 to avoid leakage
- supported provider type is currently `openai_compatible`
- `agent.review.prediction_origins` may contain only `model` and/or `agent`
- agent workflows read and write structured artifacts; they do not bypass the run contract
