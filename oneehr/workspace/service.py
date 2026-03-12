from __future__ import annotations

import json
from dataclasses import dataclass
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
from oneehr.artifacts.read import RunManifest, read_run_manifest
from oneehr.cases.bundle import (
    list_cases as _list_cases,
    read_case as _read_case,
    read_cases_index as _read_cases_index,
)


def open_run_workspace(run_root: str | Path) -> RunWorkspace:
    root = Path(run_root).resolve()
    manifest = read_run_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"Missing run_manifest.json under {root}")
    return RunWorkspace(run_root=root, manifest=manifest)


@dataclass(frozen=True)
class RunWorkspace:
    run_root: Path
    manifest: RunManifest

    @property
    def run_name(self) -> str:
        return self.run_root.name

    def read_json(self, relative_path: str | Path) -> dict[str, Any]:
        path = self.run_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing JSON artifact: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_optional_json(self, relative_path: str | Path) -> dict[str, Any]:
        path = self.run_root / relative_path
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def training_summary(self) -> dict[str, Any]:
        return self.read_optional_json("summary.json")

    def training_records(self) -> list[dict[str, Any]]:
        return _ensure_records(self.training_summary())

    def analysis_index(self) -> dict[str, Any]:
        return _read_analysis_index(self.run_root)

    def analysis_index_optional(self) -> dict[str, Any]:
        try:
            return self.analysis_index()
        except FileNotFoundError:
            return {}

    def analysis_modules(self) -> list[str]:
        try:
            return _list_analysis_modules(self.run_root)
        except FileNotFoundError:
            return []

    def analysis_summary(self, module_name: str) -> dict[str, Any]:
        return _read_analysis_summary(self.run_root, module_name)

    def analysis_table(self, module_name: str, table_name: str) -> pd.DataFrame:
        return _read_analysis_table(self.run_root, module_name, table_name)

    def analysis_plot_spec(self, module_name: str, plot_name: str) -> dict[str, Any]:
        return _read_analysis_plot_spec(self.run_root, module_name, plot_name)

    def failure_case_artifacts(self, module_name: str = "prediction_audit") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in _list_failure_case_paths(self.run_root, module_name=module_name):
            df = pd.read_parquet(path)
            rows.append(
                {
                    "module": str(module_name),
                    "name": path.stem,
                    "path": str(path.relative_to(self.run_root)),
                    "row_count": int(len(df)),
                    "columns": [str(col) for col in df.columns],
                    "patient_count": int(df["patient_id"].astype(str).nunique()) if "patient_id" in df.columns else 0,
                }
            )
        return rows

    def failure_case_rows(
        self,
        module_name: str = "prediction_audit",
        *,
        name: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        df = _read_failure_cases(self.run_root, module_name=module_name, name=name)
        if limit is not None:
            df = df.head(int(limit)).reset_index(drop=True)
        return {
            "module": str(module_name),
            "name": None if name is None else str(name),
            "row_count": int(len(df)),
            "columns": [str(col) for col in df.columns],
            "records": df.to_dict(orient="records"),
        }

    def patient_case_matches(
        self,
        patient_id: str,
        module_name: str = "prediction_audit",
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        payload = _describe_patient_case(self.run_root, patient_id, module_name=module_name)
        matches = list(payload.get("matches", []))
        if limit is not None:
            matches = matches[: int(limit)]
        return {
            "module": str(module_name),
            "patient_id": str(patient_id),
            "n_matches": int(payload.get("n_matches", len(matches))),
            "matches": matches,
        }

    def cases_index(self) -> dict[str, Any]:
        return _read_cases_index(self.run_root)

    def cases_index_optional(self) -> dict[str, Any]:
        try:
            return self.cases_index()
        except FileNotFoundError:
            return {}

    def case_records(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return _list_cases(self.run_root, limit=limit)

    def read_case(self, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
        return _read_case(self.run_root, case_id, limit=limit)

    def agent_predict_summary(self) -> dict[str, Any]:
        return self.read_json(Path("agent") / "predict" / "summary.json")

    def agent_predict_summary_optional(self) -> dict[str, Any]:
        return self.read_optional_json(Path("agent") / "predict" / "summary.json")

    def agent_predict_records(self) -> list[dict[str, Any]]:
        return _ensure_records(self.agent_predict_summary_optional())

    def agent_review_summary(self) -> dict[str, Any]:
        return self.read_json(Path("agent") / "review" / "summary.json")

    def agent_review_summary_optional(self) -> dict[str, Any]:
        return self.read_optional_json(Path("agent") / "review" / "summary.json")

    def agent_review_records(self) -> list[dict[str, Any]]:
        return _ensure_records(self.agent_review_summary_optional())

    def describe(self) -> dict[str, Any]:
        train_records = self.training_records()
        agent_predict_records = self.agent_predict_records()
        agent_review_records = self.agent_review_records()
        analysis_index = self.analysis_index_optional()
        cases_index = self.cases_index_optional()
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
            "run_name": self.run_name,
            "run_dir": str(self.run_root),
            "manifest": {
                "schema_version": int(self.manifest.schema_version),
                "task": dict((self.manifest.data.get("task") or {})),
                "split": dict((self.manifest.data.get("split") or {})),
                "cases": dict((self.manifest.data.get("cases") or {})),
                "agent": dict((self.manifest.data.get("agent") or {})),
            },
            "training": {
                "record_count": int(len(train_records)),
                "models": sorted({str(rec.get("model")) for rec in train_records if rec.get("model") is not None}),
                "splits": sorted({str(rec.get("split")) for rec in train_records if rec.get("split") is not None}),
                "summary_path": _relative_path_or_none(self.run_root / "summary.json", base=self.run_root),
            },
            "analysis": {
                "has_index": bool(analysis_index),
                "modules": modules,
                "index_path": _relative_path_or_none(self.run_root / "analysis" / "index.json", base=self.run_root),
            },
            "cases": {
                "case_count": int(cases_index.get("case_count", 0) or 0),
                "index_path": _relative_path_or_none(self.run_root / "cases" / "index.json", base=self.run_root),
            },
            "agent_predict": {
                "record_count": int(len(agent_predict_records)),
                "predictors": sorted(
                    {
                        str(rec.get("predictor_name"))
                        for rec in agent_predict_records
                        if rec.get("predictor_name") is not None
                    }
                ),
                "summary_path": _relative_path_or_none(
                    self.run_root / "agent" / "predict" / "summary.json",
                    base=self.run_root,
                ),
            },
            "agent_review": {
                "record_count": int(len(agent_review_records)),
                "reviewers": sorted(
                    {
                        str(rec.get("reviewer_name"))
                        for rec in agent_review_records
                        if rec.get("reviewer_name") is not None
                    }
                ),
                "summary_path": _relative_path_or_none(
                    self.run_root / "agent" / "review" / "summary.json",
                    base=self.run_root,
                ),
            },
            "artifacts": {
                "has_models_dir": bool((self.run_root / "models").exists()),
                "has_preds_dir": bool((self.run_root / "preds").exists()),
                "has_splits_dir": bool((self.run_root / "splits").exists()),
                "has_cases_dir": bool((self.run_root / "cases").exists()),
                "has_agent_dir": bool((self.run_root / "agent").exists()),
            },
        }


@dataclass(frozen=True)
class WorkspaceStore:
    root_dir: Path

    def __init__(self, root_dir: str | Path):
        object.__setattr__(self, "root_dir", Path(root_dir).resolve())

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root_dir.iterdir(), key=lambda p: p.name):
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
                    "has_cases_index": bool((path / "cases" / "index.json").exists()),
                    "has_agent_predict_summary": bool((path / "agent" / "predict" / "summary.json").exists()),
                    "has_agent_review_summary": bool((path / "agent" / "review" / "summary.json").exists()),
                    "mtime_unix": float(path.stat().st_mtime),
                }
            )
        return rows

    def open_run(self, run_name: str) -> RunWorkspace:
        run_root = (self.root_dir / run_name).resolve()
        manifest = read_run_manifest(run_root)
        if manifest is None:
            raise FileNotFoundError(f"Missing run under {self.root_dir}: {run_name}")
        return RunWorkspace(run_root=run_root, manifest=manifest)


def _ensure_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return [row for row in records if isinstance(row, dict)]


def _relative_path_or_none(path: Path, *, base: Path) -> str | None:
    if not path.exists():
        return None
    return str(path.relative_to(base))
