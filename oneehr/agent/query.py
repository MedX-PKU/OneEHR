from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.analysis.read import (
    describe_patient_case as _describe_patient_case,
    list_analysis_modules as _list_analysis_modules,
    list_failure_case_paths as _list_failure_case_paths,
    read_analysis_index as _read_analysis_index,
    read_analysis_plot_spec as _read_analysis_plot_spec,
    read_analysis_summary as _read_analysis_summary,
    read_analysis_table as _read_analysis_table,
    read_failure_cases as _read_failure_cases,
)
from oneehr.artifacts.read import read_run_manifest
from oneehr.agent.primitives import (
    collect_case_evidence as _collect_case_evidence,
    get_case_predictions as _get_case_predictions,
    get_patient_static as _get_patient_static,
    get_patient_timeline as _get_patient_timeline,
    render_case_prompt as _render_case_prompt,
)
from oneehr.agent.workspace import (
    list_workspace_cases as _list_workspace_cases,
    read_workspace_case as _read_workspace_case,
    read_workspace_index as _read_workspace_index,
)
from oneehr.llm.templates import describe_prompt_template as _describe_prompt_template
from oneehr.llm.templates import list_prompt_templates as _list_prompt_templates


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        manifest = read_run_manifest(path)
        if manifest is None:
            continue
        rows.append(
            {
                "run_name": path.name,
                "run_dir": str(path),
                "schema_version": int(manifest.schema_version),
                "task": dict((manifest.data.get("task") or {})),
                "split": dict((manifest.data.get("split") or {})),
                "has_train_summary": bool((path / "summary.json").exists()),
                "has_analysis_index": bool((path / "analysis" / "index.json").exists()),
                "has_llm_summary": bool((path / "llm" / "summary.json").exists()),
                "has_workspace_index": bool((path / "workspace" / "index.json").exists()),
                "mtime_unix": float(path.stat().st_mtime),
            }
        )
    return rows


def list_prompt_templates(*, family: str | None = None) -> list[dict[str, Any]]:
    return _list_prompt_templates(family=family)


def describe_prompt_template(name: str) -> dict[str, Any]:
    return _describe_prompt_template(name)


def read_workspace_index(run_root: str | Path) -> dict[str, Any]:
    return _read_workspace_index(run_root)


def list_workspace_cases(run_root: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    return _list_workspace_cases(run_root, limit=limit)


def read_workspace_case(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    return _read_workspace_case(run_root, case_id, limit=limit)


def get_patient_timeline(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    return _get_patient_timeline(run_root, case_id, limit=limit)


def get_patient_static(run_root: str | Path, case_id: str) -> dict[str, Any]:
    return _get_patient_static(run_root, case_id)


def get_case_predictions(
    run_root: str | Path,
    case_id: str,
    *,
    source: str | None = None,
    model_name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    return _get_case_predictions(run_root, case_id, source=source, model_name=model_name, limit=limit)


def collect_case_evidence(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    return _collect_case_evidence(run_root, case_id, limit=limit)


def render_case_prompt(
    *,
    cfg,
    run_root: str | Path,
    case_id: str,
    template_name: str | None = None,
    source: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    return _render_case_prompt(
        cfg=cfg,
        run_root=run_root,
        case_id=case_id,
        template_name=template_name,
        source=source,
        model_name=model_name,
    )


def describe_run(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    manifest = read_run_manifest(run_root)
    if manifest is None:
        raise FileNotFoundError(f"Missing run_manifest.json under {run_root}")

    train_summary = _read_json(run_root / "summary.json")
    llm_summary = _read_json(run_root / "llm" / "summary.json")
    analysis_index = _read_json(run_root / "analysis" / "index.json")

    train_records = _ensure_records(train_summary)
    llm_records = _ensure_records(llm_summary)
    modules = []
    if isinstance(analysis_index.get("modules"), list):
        modules = [
            {
                "name": str(item.get("name")),
                "status": str(item.get("status")),
                "summary_path": item.get("summary_path"),
            }
            for item in analysis_index["modules"]
            if isinstance(item, dict)
        ]

    return {
        "run_name": run_root.name,
        "run_dir": str(run_root),
        "manifest": {
            "schema_version": int(manifest.schema_version),
            "task": dict((manifest.data.get("task") or {})),
            "split": dict((manifest.data.get("split") or {})),
            "llm": dict((manifest.data.get("llm") or {})),
            "workspace": dict((manifest.data.get("workspace") or {})),
        },
        "training": {
            "record_count": int(len(train_records)),
            "models": sorted({str(rec.get("model")) for rec in train_records if rec.get("model") is not None}),
            "splits": sorted({str(rec.get("split")) for rec in train_records if rec.get("split") is not None}),
            "summary_path": None if not train_summary else str((run_root / "summary.json").relative_to(run_root)),
        },
        "analysis": {
            "has_index": bool(analysis_index),
            "modules": modules,
            "index_path": None if not analysis_index else str((run_root / "analysis" / "index.json").relative_to(run_root)),
        },
        "llm": {
            "record_count": int(len(llm_records)),
            "models": sorted({str(rec.get("llm_model")) for rec in llm_records if rec.get("llm_model") is not None}),
            "summary_path": None if not llm_summary else str((run_root / "llm" / "summary.json").relative_to(run_root)),
        },
        "artifacts": {
            "has_models_dir": bool((run_root / "models").exists()),
            "has_preds_dir": bool((run_root / "preds").exists()),
            "has_splits_dir": bool((run_root / "splits").exists()),
            "has_workspace_dir": bool((run_root / "workspace").exists()),
        },
    }


def list_analysis_modules(run_root: str | Path) -> list[str]:
    return _list_analysis_modules(run_root)


def read_analysis_index(run_root: str | Path) -> dict[str, Any]:
    return _read_analysis_index(run_root)


def read_analysis_summary(run_root: str | Path, module_name: str) -> dict[str, Any]:
    return _read_analysis_summary(run_root, module_name)


def read_analysis_table(
    run_root: str | Path,
    module_name: str,
    table_name: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    df = _read_analysis_table(run_root, module_name, table_name)
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
    return _read_analysis_plot_spec(run_root, module_name, plot_name)


def list_failure_cases(run_root: str | Path, module_name: str = "prediction_audit") -> list[dict[str, Any]]:
    run_root = Path(run_root)
    rows: list[dict[str, Any]] = []
    for path in _list_failure_case_paths(run_root, module_name=module_name):
        df = pd.read_parquet(path)
        rows.append(
            {
                "module": str(module_name),
                "name": path.stem,
                "path": str(path.relative_to(run_root)),
                "row_count": int(len(df)),
                "columns": [str(col) for col in df.columns],
                "patient_count": int(df["patient_id"].astype(str).nunique()) if "patient_id" in df.columns else 0,
            }
        )
    return rows


def read_failure_cases(
    run_root: str | Path,
    module_name: str = "prediction_audit",
    *,
    name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    df = _read_failure_cases(run_root, module_name=module_name, name=name)
    if limit is not None:
        df = df.head(int(limit)).reset_index(drop=True)
    return {
        "module": str(module_name),
        "name": None if name is None else str(name),
        "row_count": int(len(df)),
        "columns": [str(col) for col in df.columns],
        "records": df.to_dict(orient="records"),
    }


def describe_patient_case(
    run_root: str | Path,
    patient_id: str,
    module_name: str = "prediction_audit",
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    payload = _describe_patient_case(run_root, patient_id, module_name=module_name)
    matches = list(payload.get("matches", []))
    if limit is not None:
        matches = matches[: int(limit)]
    return {
        "module": str(module_name),
        "patient_id": str(patient_id),
        "n_matches": int(payload.get("n_matches", len(matches))),
        "matches": matches,
    }


def compare_cohorts(
    run_root: str | Path,
    *,
    split: str | None = None,
    left_role: str = "train",
    right_role: str = "test",
    top_k: int = 10,
) -> dict[str, Any]:
    split_roles = _read_analysis_table(run_root, "cohort_analysis", "split_roles")
    drift = _read_analysis_table(run_root, "cohort_analysis", "feature_drift")
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return [rec for rec in records if isinstance(rec, dict)]


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
