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
from oneehr.eval.query import (
    read_eval_index as _read_eval_index,
    read_eval_report_summary as _read_eval_report_summary,
    read_eval_summary as _read_eval_summary,
    read_eval_table_frame as _read_eval_table_frame,
)
from oneehr.artifacts.read import RunManifest, read_run_manifest


def open_run_view(run_root: str | Path) -> RunView:
    root = Path(run_root).resolve()
    manifest = read_run_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"Missing run_manifest.json under {root}")
    return RunView(run_root=root, manifest=manifest)


@dataclass(frozen=True)
class RunView:
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

    def testing_summary(self) -> dict[str, Any]:
        return self.read_optional_json(Path("test_runs") / "test_summary.json")

    def testing_records(self) -> list[dict[str, Any]]:
        return _ensure_records(self.testing_summary())

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

    def eval_index(self) -> dict[str, Any]:
        return _read_eval_index(self.run_root)

    def eval_index_optional(self) -> dict[str, Any]:
        try:
            return self.eval_index()
        except FileNotFoundError:
            return {}

    def eval_summary(self) -> dict[str, Any]:
        return _read_eval_summary(self.run_root)

    def eval_summary_optional(self) -> dict[str, Any]:
        try:
            return self.eval_summary()
        except FileNotFoundError:
            return {}

    def eval_report_summary(self) -> dict[str, Any]:
        return _read_eval_report_summary(self.run_root)

    def eval_report_summary_optional(self) -> dict[str, Any]:
        try:
            return self.eval_report_summary()
        except FileNotFoundError:
            return {}

    def eval_table(self, table_name: str) -> pd.DataFrame:
        return _read_eval_table_frame(self.run_root, table_name=table_name)

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

    def describe(self) -> dict[str, Any]:
        train_records = self.training_records()
        test_records = self.testing_records()
        analysis_index = self.analysis_index_optional()
        eval_index = self.eval_index_optional()
        eval_summary = self.eval_summary_optional()
        eval_report = self.eval_report_summary_optional()
        testing_snapshot = _testing_summary_snapshot(
            test_records,
            task=self.manifest.data.get("task") or {},
        )
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
            },
            "training": {
                "record_count": int(len(train_records)),
                "models": sorted({str(rec.get("model")) for rec in train_records if rec.get("model") is not None}),
                "splits": sorted({str(rec.get("split")) for rec in train_records if rec.get("split") is not None}),
                "summary_path": _relative_path_or_none(self.run_root / "summary.json", base=self.run_root),
            },
            "testing": {
                **testing_snapshot,
                "summary_path": _relative_path_or_none(
                    self.run_root / "test_runs" / "test_summary.json",
                    base=self.run_root,
                ),
            },
            "analysis": {
                "has_index": bool(analysis_index),
                "modules": modules,
                "index_path": _relative_path_or_none(self.run_root / "analysis" / "index.json", base=self.run_root),
            },
            "eval": {
                "instance_count": int(eval_index.get("instance_count", 0) or 0),
                "system_count": int(len(_ensure_records(eval_summary))),
                "leaderboard_rows": int(eval_report.get("leaderboard_rows", 0) or 0),
                "primary_metric": eval_report.get("primary_metric"),
                "index_path": _relative_path_or_none(self.run_root / "eval" / "index.json", base=self.run_root),
                "summary_path": _relative_path_or_none(self.run_root / "eval" / "summary.json", base=self.run_root),
                "report_summary_path": _relative_path_or_none(
                    self.run_root / "eval" / "reports" / "summary.json",
                    base=self.run_root,
                ),
            },
            "artifacts": {
                "has_models_dir": bool((self.run_root / "models").exists()),
                "has_preds_dir": bool((self.run_root / "preds").exists()),
                "has_splits_dir": bool((self.run_root / "splits").exists()),
                "has_eval_dir": bool((self.run_root / "eval").exists()),
            },
        }


@dataclass(frozen=True)
class RunCatalog:
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
                    "has_test_summary": bool((path / "test_runs" / "test_summary.json").exists()),
                    "has_analysis_index": bool((path / "analysis" / "index.json").exists()),
                    "has_eval_index": bool((path / "eval" / "index.json").exists()),
                    "has_eval_summary": bool((path / "eval" / "summary.json").exists()),
                    "has_eval_report_summary": bool((path / "eval" / "reports" / "summary.json").exists()),
                    "testing": {
                        **_testing_summary_snapshot(
                            _ensure_records(_read_top_summary(path / "test_runs" / "test_summary.json")),
                            task=manifest.data.get("task") or {},
                        ),
                        "summary_path": _relative_path_or_none(path / "test_runs" / "test_summary.json", base=path),
                    },
                    "mtime_unix": float(path.stat().st_mtime),
                }
            )
        return rows

    def open_run(self, run_name: str) -> RunView:
        run_root = (self.root_dir / run_name).resolve()
        manifest = read_run_manifest(run_root)
        if manifest is None:
            raise FileNotFoundError(f"Missing run under {self.root_dir}: {run_name}")
        return RunView(run_root=run_root, manifest=manifest)


def _ensure_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return [row for row in records if isinstance(row, dict)]


def _relative_path_or_none(path: Path, *, base: Path) -> str | None:
    if not path.exists():
        return None
    return str(path.relative_to(base))


def _testing_summary_snapshot(records: list[dict[str, Any]], *, task: dict[str, Any]) -> dict[str, Any]:
    primary_metric = "auroc" if str(task.get("kind", "binary")) == "binary" else "rmse"
    best_model = _best_test_metric_model(records, primary_metric=primary_metric)
    return {
        "record_count": int(len(records)),
        "models": sorted({str(rec.get("model")) for rec in records if rec.get("model") is not None}),
        "splits": sorted({str(rec.get("split")) for rec in records if rec.get("split") is not None}),
        "primary_metric": primary_metric,
        "best_model": best_model,
        "best_score": None if best_model is None else float(best_model["value"]),
    }


def _best_test_metric_model(records: list[dict[str, Any]], *, primary_metric: str) -> dict[str, Any] | None:
    if not records:
        return None
    rows = pd.DataFrame(records)
    if rows.empty or "model" not in rows.columns or primary_metric not in rows.columns:
        return None
    rows[primary_metric] = pd.to_numeric(rows[primary_metric], errors="coerce")
    rows = rows.dropna(subset=[primary_metric]).reset_index(drop=True)
    if rows.empty:
        return None
    grouped = rows.groupby("model", sort=True)[primary_metric].mean().reset_index(name="mean_metric")
    ascending = primary_metric == "rmse"
    best = grouped.sort_values("mean_metric", ascending=ascending, kind="stable").iloc[0]
    return {
        "model": str(best["model"]),
        "metric": primary_metric,
        "value": float(best["mean_metric"]),
    }


def _read_top_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
