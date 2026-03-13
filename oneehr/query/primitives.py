from __future__ import annotations

from pathlib import Path
from typing import Any

from oneehr.agent.cases import build_case_context, case_static_features, select_case_predictions
from oneehr.config.schema import ExperimentConfig
from oneehr.agent.render import render_prompt
from oneehr.agent.predict_schema import schema_prompt_text
from oneehr.agent.templates import get_prompt_template
from oneehr.agent.review_schema import review_schema_prompt_text
from oneehr.runview import open_run_view


def get_case_timeline(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    case = open_run_view(run_root).read_case(case_id, limit=limit)
    events = list(case.get("events", []))
    return {
        "case_id": str(case_id),
        "patient_id": str(case.get("patient_id")),
        "row_count": int(len(events)),
        "records": events,
    }


def get_case_static(run_root: str | Path, case_id: str) -> dict[str, Any]:
    case = open_run_view(run_root).read_case(case_id)
    features = case_static_features(case)
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
    case = open_run_view(run_root).read_case(case_id, limit=limit)
    rows = select_case_predictions(case, origin=origin, predictor_name=predictor_name)
    return {
        "case_id": str(case_id),
        "patient_id": str(case.get("patient_id")),
        "row_count": int(len(rows)),
        "records": rows.to_dict(orient="records"),
    }


def collect_case_evidence(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    case = open_run_view(run_root).read_case(case_id, limit=limit)
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
    case = open_run_view(run_root).read_case(case_id)
    template = get_prompt_template(template_name or cfg.agent.predict.prompt_template)

    if template.family == "prediction":
        context = build_case_context(
            case,
            default_prediction_mode=str(cfg.task.prediction_mode),
            include_ground_truth=cfg.agent.predict.prompt.include_labels_context,
        )
        prompt = render_prompt(
            cfg=cfg,
            instance=context.instance,
            dynamic=context.dynamic,
            static_row=context.static_row,
            schema_text=schema_prompt_text(
                task_kind=cfg.task.kind,
                include_explanation=cfg.agent.predict.output.include_explanation,
                include_confidence=cfg.agent.predict.output.include_confidence,
            ),
            template_name=template.name,
        )
        return {"template": template.name, "family": template.family, "prompt": prompt}

    if template.family == "review":
        context = build_case_context(
            case,
            default_prediction_mode=str(cfg.task.prediction_mode),
            include_ground_truth=cfg.agent.review.prompt.include_ground_truth,
        )
        preds = select_case_predictions(case, origin=origin, predictor_name=predictor_name)
        if preds.empty:
            raise ValueError("No matching case prediction found for review prompt rendering.")
        if len(preds) > 1:
            raise ValueError("Multiple matching case predictions found. Use --origin and/or --predictor-name.")
        prompt = render_prompt(
            cfg=cfg,
            instance=context.instance,
            dynamic=context.dynamic,
            static_row=context.static_row,
            schema_text=review_schema_prompt_text(),
            template_name=template.name,
            prompt_cfg=cfg.agent.review.prompt,
            target_prediction=preds.iloc[0].to_dict(),
            analysis_refs=case.get("analysis_refs"),
        )
        return {"template": template.name, "family": template.family, "prompt": prompt}

    raise ValueError(f"Unsupported template family {template.family!r}")
