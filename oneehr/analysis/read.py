from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


def read_analysis_index(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    path = run_root / "analysis" / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis index: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_analysis_modules(run_root: str | Path) -> list[str]:
    index = read_analysis_index(run_root)
    modules = index.get("modules", [])
    return [str(item["name"]) for item in modules if isinstance(item, dict) and "name" in item]


def read_analysis_summary(run_root: str | Path, module_name: str) -> dict[str, Any]:
    run_root = Path(run_root)
    path = run_root / "analysis" / module_name / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis summary for {module_name!r}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_analysis_table(run_root: str | Path, module_name: str, table_name: str) -> pd.DataFrame:
    run_root = Path(run_root)
    path = run_root / "analysis" / module_name / f"{table_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis table {table_name!r} for {module_name!r}: {path}")
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def read_analysis_plot_spec(run_root: str | Path, module_name: str, plot_name: str) -> dict[str, Any]:
    run_root = Path(run_root)
    path = run_root / "analysis" / module_name / "plots" / f"{plot_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis plot {plot_name!r} for {module_name!r}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_failure_case_paths(run_root: str | Path, module_name: str = "prediction_audit") -> list[Path]:
    run_root = Path(run_root)
    case_dir = run_root / "analysis" / module_name / "cases"
    if not case_dir.exists():
        return []
    return sorted(case_dir.glob("*.parquet"))


def read_failure_cases(
    run_root: str | Path,
    module_name: str = "prediction_audit",
    *,
    name: str | None = None,
) -> pd.DataFrame:
    paths = list_failure_case_paths(run_root, module_name=module_name)
    if name is not None:
        paths = [path for path in paths if path.stem == name]
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def describe_patient_case(
    run_root: str | Path,
    patient_id: str,
    module_name: str = "prediction_audit",
) -> dict[str, Any]:
    cases = read_failure_cases(run_root, module_name=module_name)
    if cases.empty or "patient_id" not in cases.columns:
        return {"patient_id": patient_id, "matches": []}
    matches = cases[cases["patient_id"].astype(str) == str(patient_id)].copy()
    return {
        "patient_id": str(patient_id),
        "n_matches": int(len(matches)),
        "matches": matches.to_dict(orient="records"),
    }
