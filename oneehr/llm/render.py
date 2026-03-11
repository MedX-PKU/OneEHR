from __future__ import annotations

import json

import pandas as pd

from oneehr.config.schema import ExperimentConfig


def render_prompt(
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
    events = _select_events(dynamic=dynamic, anchor_time=anchor_time, cfg=cfg)

    if "patient_profile" in names:
        sections.append(_render_patient_profile(instance=instance, static_row=static_row))
    if "event_timeline" in names:
        sections.append(_render_event_timeline(events=events))
    if "code_summary" in names:
        sections.append(_render_code_summary(events=events))
    if "prediction_task" in names:
        sections.append(_render_prediction_task(cfg=cfg, instance=instance))
    if "output_schema" in names:
        sections.append(_render_output_schema(schema_text=schema_text))

    return "\n\n".join(sec for sec in sections if sec.strip()).strip() + "\n"


def _anchor_time(instance: dict[str, object]) -> pd.Timestamp | None:
    raw = instance.get("bin_time")
    if raw in {None, "", "NaT"}:
        return None
    return pd.to_datetime(raw, errors="raise")


def _select_events(
    *,
    dynamic: pd.DataFrame | None,
    anchor_time: pd.Timestamp | None,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    if dynamic is None or dynamic.empty:
        return pd.DataFrame(columns=["event_time", "code", "value"])

    events = dynamic.copy()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="raise")
    if anchor_time is not None:
        events = events[events["event_time"] <= anchor_time].copy()

    if cfg.llm.prompt.history_window is not None and anchor_time is not None:
        start = anchor_time - pd.to_timedelta(cfg.llm.prompt.history_window)
        events = events[events["event_time"] >= start].copy()

    events = events.sort_values("event_time", kind="stable")
    if len(events) > cfg.llm.prompt.max_events:
        events = events.tail(cfg.llm.prompt.max_events)
    if cfg.llm.prompt.time_order == "desc":
        events = events.iloc[::-1].reset_index(drop=True)
    else:
        events = events.reset_index(drop=True)
    return events


def _render_patient_profile(*, instance: dict[str, object], static_row: pd.Series | None) -> str:
    lines = ["Patient Profile", f"- patient_id: {instance['patient_id']}"]
    if instance.get("bin_time") not in {None, "", "NaT"}:
        lines.append(f"- anchor_time: {pd.to_datetime(instance['bin_time']).isoformat()}")

    if static_row is None or static_row.empty:
        lines.append("- static_features: none")
        return "\n".join(lines)

    for col, value in static_row.items():
        if str(col) == "patient_id":
            continue
        if pd.isna(value):
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

    counts = (
        events.groupby("code", sort=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
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


def _render_prediction_task(*, cfg: ExperimentConfig, instance: dict[str, object]) -> str:
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


def _render_output_schema(*, schema_text: str) -> str:
    return "Output Schema\n" + schema_text + "\n\nReturn exactly one JSON object matching the schema above."

