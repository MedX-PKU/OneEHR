from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.query.primitives import (
    collect_case_evidence as _collect_case_evidence,
    get_case_predictions as _get_case_predictions,
    get_case_static as _get_case_static,
    get_case_timeline as _get_case_timeline,
    render_case_prompt as _render_case_prompt,
)
from oneehr.agent.templates import describe_prompt_template as _describe_prompt_template
from oneehr.agent.templates import list_prompt_templates as _list_prompt_templates
from oneehr.workspace import WorkspaceStore, open_run_workspace


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    return WorkspaceStore(root).list_runs()


def list_prompt_templates(*, family: str | None = None) -> list[dict[str, Any]]:
    return _list_prompt_templates(family=family)


def describe_prompt_template(name: str) -> dict[str, Any]:
    return _describe_prompt_template(name)


def read_agent_predict_summary(run_root: str | Path) -> dict[str, Any]:
    return open_run_workspace(run_root).agent_predict_summary()


def read_agent_review_summary(run_root: str | Path) -> dict[str, Any]:
    return open_run_workspace(run_root).agent_review_summary()


def read_cases_index(run_root: str | Path) -> dict[str, Any]:
    return open_run_workspace(run_root).cases_index()


def list_cases(run_root: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    return open_run_workspace(run_root).case_records(limit=limit)


def read_case(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    return open_run_workspace(run_root).read_case(case_id, limit=limit)


def get_case_timeline(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    return _get_case_timeline(run_root, case_id, limit=limit)


def get_case_static(run_root: str | Path, case_id: str) -> dict[str, Any]:
    return _get_case_static(run_root, case_id)


def get_case_predictions(
    run_root: str | Path,
    case_id: str,
    *,
    origin: str | None = None,
    predictor_name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    return _get_case_predictions(
        run_root,
        case_id,
        origin=origin,
        predictor_name=predictor_name,
        limit=limit,
    )


def collect_case_evidence(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    return _collect_case_evidence(run_root, case_id, limit=limit)


def render_case_prompt(
    *,
    cfg,
    run_root: str | Path,
    case_id: str,
    template_name: str | None = None,
    origin: str | None = None,
    predictor_name: str | None = None,
) -> dict[str, Any]:
    return _render_case_prompt(
        cfg=cfg,
        run_root=run_root,
        case_id=case_id,
        template_name=template_name,
        origin=origin,
        predictor_name=predictor_name,
    )


def describe_run(run_root: str | Path) -> dict[str, Any]:
    return open_run_workspace(run_root).describe()


def list_analysis_modules(run_root: str | Path) -> list[str]:
    return open_run_workspace(run_root).analysis_modules()


def read_analysis_index(run_root: str | Path) -> dict[str, Any]:
    return open_run_workspace(run_root).analysis_index()


def read_analysis_summary(run_root: str | Path, module_name: str) -> dict[str, Any]:
    return open_run_workspace(run_root).analysis_summary(module_name)


def read_analysis_table(
    run_root: str | Path,
    module_name: str,
    table_name: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    df = open_run_workspace(run_root).analysis_table(module_name, table_name)
    if limit is not None:
        df = df.head(int(limit)).reset_index(drop=True)
    return {
        "module": str(module_name),
        "table": str(table_name),
        "row_count": int(len(df)),
        "columns": [str(col) for col in df.columns],
        "records": df.to_dict(orient="records"),
    }


def read_analysis_plot_spec(run_root: str | Path, module_name: str, plot_name: str) -> dict[str, Any]:
    return open_run_workspace(run_root).analysis_plot_spec(module_name, plot_name)


def list_failure_cases(run_root: str | Path, module_name: str = "prediction_audit") -> list[dict[str, Any]]:
    return open_run_workspace(run_root).failure_case_artifacts(module_name)


def read_failure_cases(
    run_root: str | Path,
    module_name: str = "prediction_audit",
    *,
    name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    return open_run_workspace(run_root).failure_case_rows(module_name, name=name, limit=limit)


def describe_patient_case(
    run_root: str | Path,
    patient_id: str,
    module_name: str = "prediction_audit",
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    return open_run_workspace(run_root).patient_case_matches(patient_id, module_name, limit=limit)


def compare_cohorts(
    run_root: str | Path,
    *,
    split: str | None = None,
    left_role: str = "train",
    right_role: str = "test",
    top_k: int = 10,
) -> dict[str, Any]:
    workspace = open_run_workspace(run_root)
    split_roles = workspace.analysis_table("cohort_analysis", "split_roles")
    drift = workspace.analysis_table("cohort_analysis", "feature_drift")
    if split_roles.empty:
        raise ValueError("cohort_analysis split_roles table is empty")

    split_name = _resolve_split_name(split_roles, split)
    left = _select_cohort_row(split_roles, split_name, left_role)
    right = _select_cohort_row(split_roles, split_name, right_role)

    common_numeric = [col for col in ("n_patients", "n_samples", "n_labeled_samples", "label_rate", "mean_events_per_patient") if col in left.index and col in right.index]
    deltas = {}
    for col in common_numeric:
        left_val = pd.to_numeric(pd.Series([left[col]]), errors="coerce").iloc[0]
        right_val = pd.to_numeric(pd.Series([right[col]]), errors="coerce").iloc[0]
        deltas[f"{col}_delta"] = None if pd.isna(left_val) or pd.isna(right_val) else float(right_val - left_val)

    drift_rows: list[dict[str, Any]] = []
    drift_available = left_role == "train" and right_role in {"val", "test"} and not drift.empty
    if drift_available:
        block = drift[(drift["split"].astype(str) == split_name) & (drift["role"].astype(str) == right_role)].copy()
        if not block.empty:
            drift_rows = block.sort_values("abs_delta", ascending=False, kind="stable").head(int(top_k)).to_dict(orient="records")

    return {
        "split": str(split_name),
        "left_role": str(left_role),
        "right_role": str(right_role),
        "left": left.to_dict(),
        "right": right.to_dict(),
        "deltas": deltas,
        "feature_drift_available": bool(drift_available),
        "top_feature_drift": drift_rows,
    }
def _resolve_split_name(split_roles: pd.DataFrame, split: str | None) -> str:
    available = split_roles["split"].astype(str).unique().tolist()
    if split is not None:
        if split not in available:
            raise ValueError(f"Unknown split {split!r}. Available: {available}")
        return str(split)
    if len(available) == 1:
        return str(available[0])
    raise ValueError(f"split is required when multiple splits are present: {available}")


def _select_cohort_row(split_roles: pd.DataFrame, split_name: str, role: str) -> pd.Series:
    block = split_roles[(split_roles["split"].astype(str) == split_name) & (split_roles["role"].astype(str) == str(role))].copy()
    if block.empty:
        raise ValueError(f"No cohort row for split={split_name!r} role={role!r}")
    return block.iloc[0]
