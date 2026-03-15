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
from oneehr.cases.bundle import (
    list_cases as _list_cases,
    read_case as _read_case,
    read_cases_index as _read_cases_index,
)


AGENT_TASK_SPECS: dict[str, dict[str, Any]] = {
    "predict": {
        "summary_path": Path("agent") / "predict" / "summary.json",
        "detail_artifact_key": "preds_parquet",
        "failure_dir": Path("agent") / "predict" / "failures",
        "actor_field": "predictor_name",
        "default_sort": ("split", "patient_id", "instance_id"),
    },
    "review": {
        "summary_path": Path("agent") / "review" / "summary.json",
        "detail_artifact_key": "parsed_parquet",
        "failure_dir": Path("agent") / "review" / "failures",
        "actor_field": "reviewer_name",
        "default_sort": ("split", "patient_id", "case_id", "target_origin", "target_predictor_name"),
    },
}


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

    def agent_task_summary(self, task_name: str) -> dict[str, Any]:
        spec = _agent_task_spec(task_name)
        return self.read_json(spec["summary_path"])

    def agent_task_summary_optional(self, task_name: str) -> dict[str, Any]:
        spec = _agent_task_spec(task_name)
        return self.read_optional_json(spec["summary_path"])

    def agent_task_records(self, task_name: str) -> list[dict[str, Any]]:
        return _ensure_records(self.agent_task_summary_optional(task_name))

    def agent_task_actors(self, task_name: str) -> list[str]:
        spec = _agent_task_spec(task_name)
        actor_field = str(spec["actor_field"])
        return sorted(
            {
                str(record.get(actor_field))
                for record in self.agent_task_records(task_name)
                if record.get(actor_field) is not None
            }
        )

    def agent_task_splits(self, task_name: str) -> list[str]:
        return sorted(
            {
                str(record.get("split"))
                for record in self.agent_task_records(task_name)
                if record.get("split") is not None
            }
        )

    def agent_task_detail_artifacts(self, task_name: str) -> list[str]:
        spec = _agent_task_spec(task_name)
        key = str(spec["detail_artifact_key"])
        seen: set[str] = set()
        paths: list[str] = []
        for record in self.agent_task_records(task_name):
            path = _agent_record_artifact_path(record, key)
            if path is None or path in seen:
                continue
            seen.add(path)
            paths.append(path)
        return paths

    def agent_task_failure_artifacts(self, task_name: str) -> list[str]:
        spec = _agent_task_spec(task_name)
        root = self.run_root / spec["failure_dir"]
        if not root.exists():
            return []
        paths: list[str] = []
        for actor_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if not actor_dir.is_dir():
                continue
            for path in sorted(actor_dir.glob("*.jsonl"), key=lambda item: item.name):
                paths.append(str(path.relative_to(self.run_root)))
        return paths

    def agent_task_detail_rows(
        self,
        task_name: str,
        *,
        actor: str | None = None,
        split: str | None = None,
        parsed_ok: bool | None = None,
        search: str | None = None,
    ) -> pd.DataFrame:
        spec = _agent_task_spec(task_name)
        actor_field = str(spec["actor_field"])
        artifact_key = str(spec["detail_artifact_key"])
        summary_records = self.agent_task_records(task_name)
        frames: list[pd.DataFrame] = []
        seen_paths: set[str] = set()
        for record in summary_records:
            path_rel = _agent_record_artifact_path(record, artifact_key)
            if path_rel is None or path_rel in seen_paths:
                continue
            seen_paths.add(path_rel)
            path = self.run_root / path_rel
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            frame = _annotate_agent_frame(frame, record=record, actor_field=actor_field)
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = _filter_agent_frame(
            df,
            actor_field=actor_field,
            actor=actor,
            split=split,
            parsed_ok=parsed_ok,
            search=search,
        )
        return _sort_dataframe_by_columns(df, columns=tuple(spec["default_sort"]))

    def agent_task_failure_rows(
        self,
        task_name: str,
        *,
        actor: str | None = None,
        split: str | None = None,
        search: str | None = None,
    ) -> pd.DataFrame:
        spec = _agent_task_spec(task_name)
        actor_field = str(spec["actor_field"])
        failure_root = self.run_root / spec["failure_dir"]
        if not failure_root.exists():
            return pd.DataFrame()

        provider_models = _agent_provider_models(self.agent_task_records(task_name), actor_field=actor_field)
        frames: list[pd.DataFrame] = []
        for actor_dir in sorted(failure_root.iterdir(), key=lambda path: path.name):
            if not actor_dir.is_dir():
                continue
            actor_name = actor_dir.name
            if actor is not None and actor_name != str(actor):
                continue
            for path in sorted(actor_dir.glob("*.jsonl"), key=lambda item: item.name):
                if split is not None and path.stem != str(split):
                    continue
                rows = _read_jsonl_rows(path)
                if not rows:
                    continue
                frame = pd.DataFrame(rows)
                if actor_field not in frame.columns:
                    frame[actor_field] = actor_name
                if "provider_model" not in frame.columns and actor_name in provider_models:
                    frame["provider_model"] = provider_models[actor_name]
                frame["failure_path"] = str(path.relative_to(self.run_root))
                frames.append(frame)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = _filter_agent_frame(
            df,
            actor_field=actor_field,
            actor=actor,
            split=split,
            parsed_ok=None,
            search=search,
        )
        return _sort_dataframe_by_columns(df, columns=("split", actor_field, "patient_id", "case_id", "instance_id"))

    def agent_predict_summary(self) -> dict[str, Any]:
        return self.agent_task_summary("predict")

    def agent_predict_summary_optional(self) -> dict[str, Any]:
        return self.agent_task_summary_optional("predict")

    def agent_predict_records(self) -> list[dict[str, Any]]:
        return self.agent_task_records("predict")

    def agent_review_summary(self) -> dict[str, Any]:
        return self.agent_task_summary("review")

    def agent_review_summary_optional(self) -> dict[str, Any]:
        return self.agent_task_summary_optional("review")

    def agent_review_records(self) -> list[dict[str, Any]]:
        return self.agent_task_records("review")

    def describe(self) -> dict[str, Any]:
        train_records = self.training_records()
        test_records = self.testing_records()
        agent_predict_records = self.agent_predict_records()
        agent_review_records = self.agent_review_records()
        analysis_index = self.analysis_index_optional()
        cases_index = self.cases_index_optional()
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
                "cases": dict((self.manifest.data.get("cases") or {})),
                "agent": dict((self.manifest.data.get("agent") or {})),
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
                "has_eval_dir": bool((self.run_root / "eval").exists()),
                "has_cases_dir": bool((self.run_root / "cases").exists()),
                "has_agent_dir": bool((self.run_root / "agent").exists()),
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
                    "has_cases_index": bool((path / "cases" / "index.json").exists()),
                    "has_agent_predict_summary": bool((path / "agent" / "predict" / "summary.json").exists()),
                    "has_agent_review_summary": bool((path / "agent" / "review" / "summary.json").exists()),
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


def _agent_task_spec(task_name: str) -> dict[str, Any]:
    try:
        return AGENT_TASK_SPECS[str(task_name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported agent task {task_name!r}") from exc


def _agent_record_artifact_path(record: dict[str, Any], key: str) -> str | None:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(key)
    if value is None:
        return None
    path = str(value).strip()
    return path or None


def _annotate_agent_frame(
    df: pd.DataFrame,
    *,
    record: dict[str, Any],
    actor_field: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for field in (actor_field, "provider_model"):
        if field not in out.columns and record.get(field) is not None:
            out[field] = record[field]
    return out


def _agent_provider_models(records: list[dict[str, Any]], *, actor_field: str) -> dict[str, str]:
    models: dict[str, str] = {}
    for record in records:
        actor = record.get(actor_field)
        provider_model = record.get("provider_model")
        if actor is None or provider_model is None:
            continue
        models[str(actor)] = str(provider_model)
    return models


def _filter_agent_frame(
    df: pd.DataFrame,
    *,
    actor_field: str,
    actor: str | None,
    split: str | None,
    parsed_ok: bool | None,
    search: str | None,
) -> pd.DataFrame:
    out = df
    if actor is not None and actor_field in out.columns:
        out = out[out[actor_field].astype(str) == str(actor)].reset_index(drop=True)
    if split is not None and "split" in out.columns:
        out = out[out["split"].astype(str) == str(split)].reset_index(drop=True)
    if parsed_ok is not None and "parsed_ok" in out.columns:
        out = out[out["parsed_ok"].astype(bool) == bool(parsed_ok)].reset_index(drop=True)
    if search is not None:
        out = _search_dataframe(out, query=search)
    return out.reset_index(drop=True)


def _search_dataframe(df: pd.DataFrame, *, query: str) -> pd.DataFrame:
    if df.empty or not str(query).strip():
        return df
    search = str(query).strip().lower()
    text = df.astype(object).where(pd.notna(df), "")
    mask = text.apply(lambda col: col.astype(str).str.lower().str.contains(search, na=False)).any(axis=1)
    return df.loc[mask].reset_index(drop=True)


def _sort_dataframe_by_columns(df: pd.DataFrame, *, columns: tuple[str, ...]) -> pd.DataFrame:
    sort_columns = [column for column in columns if column in df.columns]
    if not sort_columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
