from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from oneehr.config.schema import ExperimentConfig


@dataclass(frozen=True)
class PromptTemplateSpec:
    name: str
    family: str
    description: str
    supported_task_kinds: tuple[str, ...]
    supported_sample_units: tuple[str, ...]
    supported_schema_versions: tuple[int, ...]
    allow_labels_context: bool = False
    default_sections: tuple[str, ...] = field(default_factory=tuple)
    section_descriptions: dict[str, str] = field(default_factory=dict)
    renderer: Callable[..., str] | None = None


_PROMPT_REGISTRY: dict[str, PromptTemplateSpec] = {}


def register_prompt_template(spec: PromptTemplateSpec) -> None:
    if spec.renderer is None:
        raise ValueError(f"Prompt template {spec.name!r} must define a renderer.")
    _PROMPT_REGISTRY[spec.name] = spec


def get_prompt_template(name: str) -> PromptTemplateSpec:
    spec = _PROMPT_REGISTRY.get(str(name))
    if spec is None:
        available = ", ".join(sorted(_PROMPT_REGISTRY))
        raise KeyError(f"Unknown prompt template {name!r}. Available: {available}")
    return spec


def list_prompt_templates(*, family: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in sorted(_PROMPT_REGISTRY):
        spec = _PROMPT_REGISTRY[name]
        if family is not None and spec.family != family:
            continue
        rows.append(describe_prompt_template(name))
    return rows


def describe_prompt_template(name: str) -> dict[str, object]:
    spec = get_prompt_template(name)
    return {
        "name": str(spec.name),
        "family": str(spec.family),
        "description": str(spec.description),
        "supported_task_kinds": list(spec.supported_task_kinds),
        "supported_sample_units": list(spec.supported_sample_units),
        "supported_schema_versions": list(spec.supported_schema_versions),
        "allow_labels_context": bool(spec.allow_labels_context),
        "default_sections": list(spec.default_sections),
        "sections": dict(spec.section_descriptions),
    }


def render_summary_v1(
    *,
    cfg: ExperimentConfig,
    instance: dict[str, object],
    dynamic: pd.DataFrame | None,
    static_row: pd.Series | None,
    schema_text: str,
) -> str:
    sections = []
    names = list(cfg.llm.prompt.sections)
    anchor_time = _anchor_time(instance)
    events = select_events(
        dynamic=dynamic,
        anchor_time=anchor_time,
        history_window=cfg.llm.prompt.history_window,
        max_events=cfg.llm.prompt.max_events,
        time_order=cfg.llm.prompt.time_order,
    )

    if "patient_profile" in names:
        sections.append(_render_patient_profile(instance=instance, static_row=static_row))
    if "event_timeline" in names:
        sections.append(_render_event_timeline(events=events))
    if "code_summary" in names:
        sections.append(_render_code_summary(events=events))
    if "prediction_task" in names:
        sections.append(_render_prediction_task(cfg=cfg))
    if "output_schema" in names:
        sections.append(_render_output_schema(schema_text=schema_text))

    return _join_sections(sections)


def render_evidence_review_v1(
    *,
    cfg: ExperimentConfig,
    instance: dict[str, object],
    dynamic: pd.DataFrame | None,
    static_row: pd.Series | None,
    schema_text: str,
    prompt_cfg,
    target_prediction: dict[str, object],
    analysis_refs: dict[str, object] | None = None,
) -> str:
    names = list(prompt_cfg.sections)
    anchor_time = _anchor_time(instance)
    events = select_events(
        dynamic=dynamic,
        anchor_time=anchor_time,
        history_window=None,
        max_events=prompt_cfg.max_events,
        time_order=prompt_cfg.time_order,
    )

    sections = []
    if "case_profile" in names:
        sections.append(_render_case_profile(instance=instance))
    if "observed_evidence" in names:
        sections.append(_render_observed_evidence(events=events, static_row=static_row, include_static=prompt_cfg.include_static))
    if "target_prediction" in names:
        sections.append(_render_target_prediction(target_prediction=target_prediction))
    if "ground_truth" in names and getattr(prompt_cfg, "include_ground_truth", False):
        sections.append(_render_ground_truth(instance=instance))
    if "analysis_context" in names and getattr(prompt_cfg, "include_analysis_context", False):
        sections.append(_render_analysis_context(analysis_refs=analysis_refs))
    if "review_rubric" in names:
        sections.append(_render_review_rubric())
    if "output_schema" in names:
        sections.append(_render_output_schema(schema_text=schema_text))
    return _join_sections(sections)


def select_events(
    *,
    dynamic: pd.DataFrame | None,
    anchor_time: pd.Timestamp | None,
    history_window: str | None,
    max_events: int,
    time_order: str,
) -> pd.DataFrame:
    if dynamic is None or dynamic.empty:
        return pd.DataFrame(columns=["event_time", "code", "value"])

    events = dynamic.copy()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="raise")
    if anchor_time is not None:
        events = events[events["event_time"] <= anchor_time].copy()
    if history_window is not None and anchor_time is not None:
        start = anchor_time - pd.to_timedelta(history_window)
        events = events[events["event_time"] >= start].copy()

    events = events.sort_values("event_time", kind="stable")
    if len(events) > int(max_events):
        events = events.tail(int(max_events))
    if str(time_order) == "desc":
        events = events.iloc[::-1].reset_index(drop=True)
    else:
        events = events.reset_index(drop=True)
    return events


def _anchor_time(instance: dict[str, object]) -> pd.Timestamp | None:
    raw = instance.get("bin_time")
    if raw in {None, "", "NaT"}:
        return None
    return pd.to_datetime(raw, errors="raise")


def _render_patient_profile(*, instance: dict[str, object], static_row: pd.Series | None) -> str:
    lines = ["Patient Profile", f"- patient_id: {instance['patient_id']}"]
    if instance.get("bin_time") not in {None, "", "NaT"}:
        lines.append(f"- anchor_time: {pd.to_datetime(instance['bin_time']).isoformat()}")

    if static_row is None or static_row.empty:
        lines.append("- static_features: none")
        return "\n".join(lines)

    for col, value in static_row.items():
        if str(col) == "patient_id" or pd.isna(value):
            continue
        lines.append(f"- {col}: {value}")
    return "\n".join(lines)


def _render_event_timeline(*, events: pd.DataFrame) -> str:
    lines = ["Event Timeline"]
    if events.empty:
        lines.append("- no events available")
        return "\n".join(lines)
    for _, row in events.iterrows():
        ts = pd.to_datetime(row["event_time"]).isoformat()
        lines.append(f"- {ts} | {row['code']} = {row['value']}")
    return "\n".join(lines)


def _render_code_summary(*, events: pd.DataFrame) -> str:
    lines = ["Code Summary"]
    if events.empty:
        lines.append("- no code summary available")
        return "\n".join(lines)

    counts = events.groupby("code", sort=False).size().sort_values(ascending=False).head(10)
    lines.append("- code_counts:")
    for code, cnt in counts.items():
        lines.append(f"  - {code}: {int(cnt)}")

    last_obs = (
        events.sort_values("event_time", kind="stable")
        .groupby("code", sort=False)
        .tail(1)
        .sort_values("event_time", kind="stable")
    )
    lines.append("- last_observations:")
    for _, row in last_obs.head(10).iterrows():
        ts = pd.to_datetime(row["event_time"]).isoformat()
        lines.append(f"  - {row['code']}: {row['value']} at {ts}")
    return "\n".join(lines)


def _render_prediction_task(*, cfg: ExperimentConfig) -> str:
    lines = ["Prediction Task"]
    if cfg.llm.sample_unit == "patient":
        lines.append("- sample_unit: patient-level")
        lines.append("- use the full observed history shown above only")
    else:
        lines.append("- sample_unit: time-window")
        lines.append("- use only information observed at or before the anchor_time")

    if cfg.task.kind == "binary":
        lines.append("- predict label as 0 or 1")
        lines.append("- also return a calibrated-looking probability in [0, 1]")
    elif cfg.task.kind == "regression":
        lines.append("- predict a numeric value")
    else:
        raise ValueError(f"Unsupported llm task kind: {cfg.task.kind!r}")

    if cfg.llm.output.include_explanation:
        lines.append("- explanation is optional but should be concise and clinically grounded")
    if cfg.llm.output.include_confidence:
        lines.append("- confidence is optional and must be in [0, 1]")
    lines.append("- return JSON only")
    return "\n".join(lines)


def _render_case_profile(*, instance: dict[str, object]) -> str:
    lines = [
        "Case Profile",
        f"- case_id: {instance['case_id']}",
        f"- patient_id: {instance['patient_id']}",
        f"- split: {instance['split']}",
        f"- prediction_mode: {instance['prediction_mode']}",
    ]
    if instance.get("bin_time") not in {None, "", "NaT"}:
        lines.append(f"- anchor_time: {pd.to_datetime(instance['bin_time']).isoformat()}")
    return "\n".join(lines)


def _render_observed_evidence(
    *,
    events: pd.DataFrame,
    static_row: pd.Series | None,
    include_static: bool,
) -> str:
    lines = ["Observed Evidence"]
    if include_static:
        lines.append("Static Features")
        if static_row is None or static_row.empty:
            lines.append("- none")
        else:
            for col, value in static_row.items():
                if str(col) == "patient_id" or pd.isna(value):
                    continue
                lines.append(f"- {col}: {value}")
    lines.append("Timeline")
    if events.empty:
        lines.append("- no events available")
        return "\n".join(lines)
    for _, row in events.iterrows():
        ts = pd.to_datetime(row["event_time"]).isoformat()
        lines.append(f"- {ts} | {row['code']} = {row['value']}")
    return "\n".join(lines)


def _render_target_prediction(*, target_prediction: dict[str, object]) -> str:
    lines = [
        "Target Prediction",
        f"- source: {target_prediction.get('source')}",
        f"- model_name: {target_prediction.get('model_name')}",
        f"- split: {target_prediction.get('split')}",
    ]
    for key in ("prediction", "probability", "value", "confidence", "parsed_ok", "error_code"):
        if target_prediction.get(key) is not None:
            lines.append(f"- {key}: {target_prediction.get(key)}")
    explanation = str(target_prediction.get("explanation") or "").strip()
    if explanation:
        lines.append("Explanation")
        lines.append(explanation)
    return "\n".join(lines)


def _render_ground_truth(*, instance: dict[str, object]) -> str:
    value = instance.get("ground_truth")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Ground Truth\n- unavailable"
    return f"Ground Truth\n- observed_outcome: {value}"


def _render_analysis_context(*, analysis_refs: dict[str, object] | None) -> str:
    lines = ["Analysis Context"]
    if not analysis_refs:
        lines.append("- no analysis references available")
        return "\n".join(lines)

    modules = analysis_refs.get("modules", [])
    if isinstance(modules, list) and modules:
        lines.append("- modules:")
        for item in modules[:10]:
            if isinstance(item, dict):
                name = item.get("name")
                status = item.get("status")
                lines.append(f"  - {name}: {status}")

    matches = analysis_refs.get("patient_case_matches", [])
    if isinstance(matches, list) and matches:
        lines.append("- patient_case_matches:")
        for item in matches[:10]:
            if isinstance(item, dict):
                lines.append(
                    f"  - {item.get('module')}::{item.get('name')} rows={item.get('row_count')}"
                )
    return "\n".join(lines)


def _render_review_rubric() -> str:
    rubric = {
        "supported": "true when the prediction/explanation is supported by the observed evidence",
        "clinically_grounded": "true when the reasoning uses clinically relevant signals rather than generic filler",
        "leakage_suspected": "true when the target appears to use information that would not be available at prediction time",
        "needs_human_review": "true when uncertainty, missing evidence, or safety concerns remain",
        "overall_score": "0 to 1 score for overall evidence quality and trustworthiness",
        "key_evidence": ["brief evidence bullets grounded in the case"],
        "missing_evidence": ["brief bullets for evidence gaps or contradictions"],
        "review_summary": "short reviewer summary grounded in the chart evidence",
    }
    return "Review Rubric\n" + json.dumps(rubric, indent=2, sort_keys=True)


def _render_output_schema(*, schema_text: str) -> str:
    return "Output Schema\n" + schema_text + "\n\nReturn exactly one JSON object matching the schema above."


def _join_sections(sections: list[str]) -> str:
    return "\n\n".join(sec for sec in sections if sec.strip()).strip() + "\n"


def safe_case_slug(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(case_id)).strip("._")
    return slug or "case"


register_prompt_template(
    PromptTemplateSpec(
        name="summary_v1",
        family="prediction",
        description="Structured prediction prompt over observed patient evidence.",
        supported_task_kinds=("binary", "regression"),
        supported_sample_units=("patient", "time"),
        supported_schema_versions=(1,),
        allow_labels_context=False,
        default_sections=(
            "patient_profile",
            "event_timeline",
            "code_summary",
            "prediction_task",
            "output_schema",
        ),
        section_descriptions={
            "patient_profile": "Patient identifier, anchor time, and optional static features.",
            "event_timeline": "Observed raw event timeline available to the model.",
            "code_summary": "Compact summary of the most frequent and latest codes.",
            "prediction_task": "Task framing and output expectations.",
            "output_schema": "Strict JSON schema contract for the model response.",
        },
        renderer=render_summary_v1,
    )
)

register_prompt_template(
    PromptTemplateSpec(
        name="evidence_review_v1",
        family="review",
        description="Structured reviewer prompt for judging evidence-grounding, leakage, and trustworthiness.",
        supported_task_kinds=("binary", "regression"),
        supported_sample_units=("patient", "time"),
        supported_schema_versions=(1,),
        allow_labels_context=True,
        default_sections=(
            "case_profile",
            "observed_evidence",
            "target_prediction",
            "ground_truth",
            "analysis_context",
            "review_rubric",
            "output_schema",
        ),
        section_descriptions={
            "case_profile": "Case identifier, patient, split, and anchor time metadata.",
            "observed_evidence": "Filtered chart evidence available at prediction time.",
            "target_prediction": "The prediction/explanation being reviewed.",
            "ground_truth": "Observed outcome, when configured for review.",
            "analysis_context": "Optional references to analysis artifacts and flagged cases.",
            "review_rubric": "Review criteria and expected reviewer behavior.",
            "output_schema": "Strict JSON schema contract for the reviewer response.",
        },
        renderer=render_evidence_review_v1,
    )
)
