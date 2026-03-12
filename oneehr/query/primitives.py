from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.config.schema import ExperimentConfig
from oneehr.agent.render import render_prompt
from oneehr.agent.predict_schema import schema_prompt_text
from oneehr.agent.templates import get_prompt_template
from oneehr.agent.review_schema import review_schema_prompt_text

from oneehr.cases.bundle import read_case


def get_case_timeline(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    case = read_case(run_root, case_id, limit=limit)
    events = list(case.get("events", []))
    return {
        "case_id": str(case_id),
        "patient_id": str(case.get("patient_id")),
        "row_count": int(len(events)),
        "records": events,
    }


def get_case_static(run_root: str | Path, case_id: str) -> dict[str, Any]:
    case = read_case(run_root, case_id)
    static_payload = case.get("static", {})
    features = {}
    if isinstance(static_payload, dict):
        maybe = static_payload.get("features", {})
        if isinstance(maybe, dict):
            features = maybe
    return {
        "case_id": str(case_id),
        "patient_id": str(case.get("patient_id")),
        "feature_count": int(len(features)),
        "features": features,
    }


def get_case_predictions(
    run_root: str | Path,
    case_id: str,
    *,
    origin: str | None = None,
    predictor_name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    case = read_case(run_root, case_id, limit=limit)
    rows = pd.DataFrame(case.get("predictions", []))
    if not rows.empty and origin is not None:
        rows = rows[rows["origin"].astype(str) == str(origin)].copy()
    if not rows.empty and predictor_name is not None:
        rows = rows[rows["predictor_name"].astype(str) == str(predictor_name)].copy()
    rows = rows.reset_index(drop=True)
    return {
        "case_id": str(case_id),
        "patient_id": str(case.get("patient_id")),
        "row_count": int(len(rows)),
        "records": rows.to_dict(orient="records"),
    }


def collect_case_evidence(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    case = read_case(run_root, case_id, limit=limit)
    return {
        "case_id": str(case_id),
        "case": {
            key: value
            for key, value in case.items()
            if key not in {"events", "predictions", "static", "analysis_refs"}
        },
        "timeline": get_case_timeline(run_root, case_id, limit=limit),
        "static": get_case_static(run_root, case_id),
        "predictions": get_case_predictions(run_root, case_id, limit=limit),
        "analysis_refs": case.get("analysis_refs", {}),
    }


def render_case_prompt(
    *,
    cfg: ExperimentConfig,
    run_root: str | Path,
    case_id: str,
    template_name: str | None = None,
    origin: str | None = None,
    predictor_name: str | None = None,
) -> dict[str, Any]:
    case = read_case(run_root, case_id)
    template = get_prompt_template(template_name or cfg.agent.predict.prompt_template)

    static_payload = case.get("static", {})
    static_features = {}
    if isinstance(static_payload, dict):
        maybe = static_payload.get("features", {})
        if isinstance(maybe, dict):
            static_features = maybe
    static_row = pd.Series({"patient_id": case.get("patient_id"), **static_features}) if static_features else None
    dynamic = pd.DataFrame(case.get("events", []))
    instance = {
        "case_id": case.get("case_id"),
        "instance_id": case.get("case_id"),
        "patient_id": case.get("patient_id"),
        "split": case.get("split"),
        "split_role": case.get("split_role", "test"),
        "bin_time": case.get("bin_time"),
        "ground_truth": case.get("ground_truth"),
        "prediction_mode": case.get("prediction_mode", cfg.task.prediction_mode),
    }

    if template.family == "prediction":
        prompt = render_prompt(
            cfg=cfg,
            instance=instance,
            dynamic=dynamic,
            static_row=static_row,
            schema_text=schema_prompt_text(
                task_kind=cfg.task.kind,
                include_explanation=cfg.agent.predict.output.include_explanation,
                include_confidence=cfg.agent.predict.output.include_confidence,
            ),
            template_name=template.name,
        )
        return {"template": template.name, "family": template.family, "prompt": prompt}

    if template.family == "review":
        preds = pd.DataFrame(case.get("predictions", []))
        if not preds.empty and origin is not None:
            preds = preds[preds["origin"].astype(str) == str(origin)].copy()
        if not preds.empty and predictor_name is not None:
            preds = preds[preds["predictor_name"].astype(str) == str(predictor_name)].copy()
        preds = preds.reset_index(drop=True)
        if preds.empty:
            raise ValueError("No matching case prediction found for review prompt rendering.")
        if len(preds) > 1:
            raise ValueError("Multiple matching case predictions found. Use --origin and/or --predictor-name.")
        prompt = render_prompt(
            cfg=cfg,
            instance=instance,
            dynamic=dynamic,
            static_row=static_row,
            schema_text=review_schema_prompt_text(),
            template_name=template.name,
            prompt_cfg=cfg.agent.review.prompt,
            target_prediction=preds.iloc[0].to_dict(),
            analysis_refs=case.get("analysis_refs"),
        )
        return {"template": template.name, "family": template.family, "prompt": prompt}

    raise ValueError(f"Unsupported template family {template.family!r}")
