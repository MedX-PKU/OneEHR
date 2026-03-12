from __future__ import annotations

from oneehr.agent.templates import PromptTemplateSpec, get_prompt_template
from oneehr.config.schema import ExperimentConfig


def validate_agent_predict_setup(cfg: ExperimentConfig) -> PromptTemplateSpec:
    predict_cfg = cfg.agent.predict
    if not predict_cfg.enabled:
        raise SystemExit("Agent prediction workflow is disabled. Set agent.predict.enabled = true in the config.")
    if cfg.task.kind not in {"binary", "regression"}:
        raise SystemExit("Agent prediction currently supports task.kind = 'binary' or 'regression' only.")
    if predict_cfg.sample_unit != cfg.task.prediction_mode:
        raise SystemExit(
            "agent.predict.sample_unit must match task.prediction_mode to keep evaluation semantics consistent."
        )
    template = _validate_agent_template(
        workflow_name="agent.predict",
        template_name=predict_cfg.prompt_template,
        expected_family="prediction",
        task_kind=cfg.task.kind,
        sample_unit=predict_cfg.sample_unit,
        json_schema_version=predict_cfg.json_schema_version,
    )
    if predict_cfg.prompt.include_labels_context and not template.allow_labels_context:
        raise SystemExit(
            "agent.predict.prompt.include_labels_context is not allowed for "
            f"prompt template {predict_cfg.prompt_template!r}."
        )
    if not predict_cfg.backends:
        raise SystemExit("At least one [[agent.predict.backends]] entry is required for agent prediction.")
    return template


def validate_agent_review_setup(cfg: ExperimentConfig) -> PromptTemplateSpec:
    review_cfg = cfg.agent.review
    if not review_cfg.enabled:
        raise SystemExit("Agent review workflow is disabled. Set agent.review.enabled = true in the config.")
    template = _validate_agent_template(
        workflow_name="agent.review",
        template_name=review_cfg.prompt_template,
        expected_family="review",
        task_kind=cfg.task.kind,
        sample_unit=cfg.task.prediction_mode,
        json_schema_version=review_cfg.json_schema_version,
    )
    if review_cfg.prompt.include_ground_truth and not template.allow_labels_context:
        raise SystemExit(
            "agent.review.prompt.include_ground_truth is not allowed for "
            f"prompt template {review_cfg.prompt_template!r}."
        )
    if not review_cfg.backends:
        raise SystemExit("At least one [[agent.review.backends]] entry is required for agent review.")
    return template


def _validate_agent_template(
    *,
    workflow_name: str,
    template_name: str,
    expected_family: str,
    task_kind: str,
    sample_unit: str,
    json_schema_version: int,
) -> PromptTemplateSpec:
    try:
        template = get_prompt_template(template_name)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    if template.family != expected_family:
        raise SystemExit(
            f"{workflow_name}.prompt_template must resolve to a {expected_family} template, "
            f"got {template.family!r}."
        )
    if task_kind not in set(template.supported_task_kinds):
        raise SystemExit(
            f"{workflow_name}.prompt_template={template_name!r} "
            f"does not support task.kind={task_kind!r}."
        )
    if sample_unit not in set(template.supported_sample_units):
        label = "prediction_mode" if workflow_name == "agent.review" else "sample_unit"
        raise SystemExit(
            f"{workflow_name}.prompt_template={template_name!r} "
            f"does not support {label}={sample_unit!r}."
        )
    if json_schema_version not in set(template.supported_schema_versions):
        raise SystemExit(
            f"{workflow_name}.prompt_template={template_name!r} "
            f"does not support json_schema_version={json_schema_version!r}."
        )
    return template
