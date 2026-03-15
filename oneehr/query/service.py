from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.agent.templates import describe_prompt_template as _describe_prompt_template
from oneehr.agent.templates import list_prompt_templates as _list_prompt_templates
from oneehr.eval.query import (
    read_eval_index as _read_eval_index,
    read_eval_report_summary as _read_eval_report_summary,
    read_eval_summary as _read_eval_summary,
    read_eval_table as _read_eval_table,
    read_instance_payload as _read_eval_instance_payload,
    read_trace_rows as _read_eval_trace_rows,
)
from oneehr.runview import RunCatalog, open_run_view


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    return RunCatalog(root).list_runs()


def list_prompt_templates(*, family: str | None = None) -> list[dict[str, Any]]:
    return _list_prompt_templates(family=family)


def describe_prompt_template(name: str) -> dict[str, Any]:
    return _describe_prompt_template(name)


def read_eval_index(run_root: str | Path) -> dict[str, Any]:
    return _read_eval_index(run_root)


def read_eval_summary(run_root: str | Path) -> dict[str, Any]:
    return _read_eval_summary(run_root)


def read_eval_report_summary(run_root: str | Path) -> dict[str, Any]:
    return _read_eval_report_summary(run_root)


def read_eval_table(
    run_root: str | Path,
    *,
    table_name: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    return _read_eval_table(run_root, table_name=table_name, limit=limit, offset=offset)


def read_eval_instance(run_root: str | Path, *, instance_id: str) -> dict[str, Any]:
    return _read_eval_instance_payload(run_root, instance_id=instance_id)


def read_eval_trace(
    run_root: str | Path,
    *,
    system_name: str,
    limit: int = 25,
    offset: int = 0,
    stage: str | None = None,
    role: str | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    return _read_eval_trace_rows(
        run_root,
        system_name=system_name,
        limit=limit,
        offset=offset,
        stage=stage,
        role=role,
        round_index=round_index,
    )


def describe_run(run_root: str | Path) -> dict[str, Any]:
    return open_run_view(run_root).describe()


def list_analysis_modules(run_root: str | Path) -> list[str]:
    return open_run_view(run_root).analysis_modules()


def read_analysis_index(run_root: str | Path) -> dict[str, Any]:
    return open_run_view(run_root).analysis_index()


def read_analysis_summary(run_root: str | Path, module_name: str) -> dict[str, Any]:
    return open_run_view(run_root).analysis_summary(module_name)


def read_analysis_table(
    run_root: str | Path,
    module_name: str,
    table_name: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    df = open_run_view(run_root).analysis_table(module_name, table_name)
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
    return open_run_view(run_root).analysis_plot_spec(module_name, plot_name)


def list_failure_cases(run_root: str | Path, module_name: str = "prediction_audit") -> list[dict[str, Any]]:
    return open_run_view(run_root).failure_case_artifacts(module_name)


def read_failure_cases(
    run_root: str | Path,
    module_name: str = "prediction_audit",
    *,
    name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    return open_run_view(run_root).failure_case_rows(module_name, name=name, limit=limit)


def describe_patient_case(
    run_root: str | Path,
    patient_id: str,
    module_name: str = "prediction_audit",
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    return open_run_view(run_root).patient_case_matches(patient_id, module_name, limit=limit)


def compare_cohorts(
    run_root: str | Path,
    *,
    split: str | None = None,
    left_role: str = "train",
    right_role: str = "test",
    top_k: int = 10,
) -> dict[str, Any]:
    run_view = open_run_view(run_root)
    split_roles = run_view.analysis_table("cohort_analysis", "split_roles")
    drift = run_view.analysis_table("cohort_analysis", "feature_drift")
    if split_roles.empty:
        raise ValueError("cohort_analysis split_roles table is empty")

    split_name = _resolve_split_name(split_roles, split)
    left = _select_cohort_row(split_roles, split_name, left_role)
    right = _select_cohort_row(split_roles, split_name, right_role)

    common_numeric = [
        col
        for col in (
            "n_patients",
            "n_samples",
            "n_labeled_samples",
            "label_rate",
            "mean_events_per_patient",
        )
        if col in left.index and col in right.index
    ]
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
    block = split_roles[
        (split_roles["split"].astype(str) == split_name) & (split_roles["role"].astype(str) == str(role))
    ].copy()
    if block.empty:
        raise ValueError(f"No cohort row for split={split_name!r} role={role!r}")
    return block.iloc[0]
