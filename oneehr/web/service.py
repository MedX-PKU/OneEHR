from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.analysis.read import read_analysis_table as _read_analysis_table_df
from oneehr.artifacts.read import read_run_manifest
from oneehr.query import (
    compare_cohorts,
    describe_patient_case,
    describe_run,
    list_failure_cases,
    list_runs,
    read_analysis_index,
    read_analysis_plot_spec,
    read_analysis_summary,
    read_failure_cases,
)
from oneehr.utils.io import as_jsonable


MODULE_META: dict[str, dict[str, str]] = {
    "dataset_profile": {
        "title": "Dataset Profile",
        "description": "Dataset scale, code distribution, and feature-space composition.",
    },
    "cohort_analysis": {
        "title": "Cohort Analysis",
        "description": "Split integrity, label-rate gaps, and top drifted features across cohort roles.",
    },
    "prediction_audit": {
        "title": "Prediction Audit",
        "description": "Model metrics, subgroup slices, and highest-error prediction cases.",
    },
    "temporal_analysis": {
        "title": "Temporal Analysis",
        "description": "Performance segmented by event density and time windows.",
    },
    "interpretability": {
        "title": "Interpretability",
        "description": "Feature-importance artifacts across model, split, and method.",
    },
    "agent_audit": {
        "title": "Agent Audit",
        "description": "Agent parse success, token usage, and failure buckets.",
    },
}

MODULE_KPI_FIELDS: dict[str, list[str]] = {
    "dataset_profile": [
        "n_dynamic_events",
        "n_patients_raw",
        "n_dynamic_features",
        "n_static_features",
        "split_count",
    ],
    "cohort_analysis": ["split_count", "role_count", "largest_label_rate_gap"],
    "prediction_audit": ["n_prediction_slices", "n_subgroup_rows"],
    "temporal_analysis": ["n_segments"],
    "interpretability": ["result_count"],
    "agent_audit": ["n_slices", "n_failure_buckets"],
}

SUMMARY_EXCLUDE_FIELDS = {
    "schema_version",
    "module",
    "status",
    "overview",
    "feature_overview",
    "reason",
    "task_kind",
    "prediction_mode",
    "primary_metric",
    "stratify_by",
    "dimensions",
    "requested_method",
}

TABLE_TITLES = {
    "top_codes": "Top Codes",
    "static_missingness": "Static Missingness",
    "feature_space": "Feature Space",
    "split_roles": "Split Roles",
    "feature_drift": "Feature Drift",
    "slices": "Slices",
    "subgroups": "Subgroups",
    "model_summary": "Model Summary",
    "segments": "Segments",
    "artifacts": "Artifacts",
    "failures": "Failures",
    "train_metrics": "Train Metric Deltas",
    "agent_predict_metrics": "Agent Predict Metric Deltas",
}


class WebUIService:
    def __init__(self, *, root_dir: str | Path):
        self.root_dir = Path(root_dir).resolve()

    def list_runs_payload(self) -> dict[str, Any]:
        rows = list_runs(self.root_dir)
        out = []
        for row in rows:
            item = dict(row)
            item["analysis_status"] = "ready" if item.get("has_analysis_index") else "pending"
            item["task_label"] = self._task_label(item.get("task"))
            item["route"] = f"/runs/{item['run_name']}"
            out.append(item)
        out.sort(key=lambda row: (row.get("mtime_unix") or 0.0), reverse=True)
        return {
            "root_dir": str(self.root_dir),
            "run_count": int(len(out)),
            "runs": out,
        }

    def describe_run_payload(self, *, run_name: str) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        run = describe_run(run_root)
        analysis = self.analysis_index_payload(run_name=run_name)
        hero = {
            "run_name": run["run_name"],
            "task_label": self._task_label((run.get("manifest") or {}).get("task")),
            "model_count": int(len((run.get("training") or {}).get("models") or [])),
            "analysis_module_count": int(len(analysis.get("modules", []))),
            "case_count": int(((run.get("cases") or {}).get("case_count")) or 0),
            "agent_predict_record_count": int(((run.get("agent_predict") or {}).get("record_count")) or 0),
            "agent_review_record_count": int(((run.get("agent_review") or {}).get("record_count")) or 0),
        }
        return {
            "run": run,
            "hero": hero,
            "analysis": analysis,
        }

    def analysis_index_payload(self, *, run_name: str) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        try:
            index = read_analysis_index(run_root)
        except FileNotFoundError:
            return {
                "run_name": str(run_name),
                "status": "missing",
                "modules": [],
                "comparison": None,
            }

        modules = []
        for item in index.get("modules", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name"))
            modules.append(self._normalize_module_index_item(run_name=run_name, item=item))

        return {
            "run_name": str(run_name),
            "status": "ok",
            "task": as_jsonable(index.get("task", {})),
            "modules": modules,
            "comparison": as_jsonable(index.get("comparison")),
        }

    def analysis_dashboard_payload(self, *, run_name: str, module_name: str) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        module_item = self._get_module_index_item(run_name=run_name, module_name=module_name)
        summary = self._read_module_summary(run_root=run_root, module_name=module_name)
        tables = self._module_table_previews(run_root=run_root, module_name=module_name, module_item=module_item)
        charts = self._module_charts(run_root=run_root, module_name=module_name, module_item=module_item)
        highlights = self._module_highlights(
            run_root=run_root,
            module_name=module_name,
            summary=summary,
            tables=tables,
        )
        case_artifacts = list_failure_cases(run_root, module_name=module_name) if self._supports_cases(module_name) else []
        drilldowns = {
            "case_artifacts": case_artifacts,
            "patient_case_supported": bool(self._supports_cases(module_name)),
        }
        return {
            "run_name": str(run_name),
            "module": {
                **module_item,
                "summary": summary,
                "kpis": self._summary_kpis(module_name=module_name, summary=summary),
            },
            "charts": charts,
            "tables": tables,
            "highlights": highlights,
            "drilldowns": drilldowns,
        }

    def analysis_table_payload(
        self,
        *,
        run_name: str,
        module_name: str,
        table_name: str,
        limit: int = 25,
        offset: int = 0,
        sort_by: str | None = None,
        sort_dir: str = "desc",
        filter_col: str | None = None,
        filter_value: str | None = None,
    ) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        self._get_module_index_item(run_name=run_name, module_name=module_name)
        df = _read_analysis_table_df(run_root, module_name, table_name)
        df = self._filter_dataframe(df, filter_col=filter_col, filter_value=filter_value)
        df = self._sort_dataframe(df, sort_by=sort_by, sort_dir=sort_dir)
        total_rows = int(len(df))
        page = df.iloc[offset : offset + limit].reset_index(drop=True)
        return {
            "run_name": str(run_name),
            "module": str(module_name),
            "table": str(table_name),
            "title": TABLE_TITLES.get(table_name, self._titleize(table_name)),
            "offset": int(offset),
            "limit": int(limit),
            "total_rows": total_rows,
            "row_count": int(len(page)),
            "columns": self._column_specs(page if not page.empty else df),
            "records": as_jsonable(page.to_dict(orient="records")),
        }

    def analysis_cases_index_payload(self, *, run_name: str, module_name: str) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        self._get_module_index_item(run_name=run_name, module_name=module_name)
        if not self._supports_cases(module_name):
            return {
                "run_name": str(run_name),
                "module": str(module_name),
                "case_artifacts": [],
            }
        return {
            "run_name": str(run_name),
            "module": str(module_name),
            "case_artifacts": list_failure_cases(run_root, module_name=module_name),
        }

    def analysis_case_rows_payload(
        self,
        *,
        run_name: str,
        module_name: str,
        case_name: str,
        limit: int = 25,
        offset: int = 0,
        sort_by: str | None = None,
        sort_dir: str = "desc",
        filter_col: str | None = None,
        filter_value: str | None = None,
    ) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        self._get_module_index_item(run_name=run_name, module_name=module_name)
        cases = list_failure_cases(run_root, module_name=module_name)
        if case_name not in {row["name"] for row in cases}:
            raise FileNotFoundError(f"Unknown case artifact {case_name!r} for module {module_name!r}")
        payload = read_failure_cases(run_root, module_name=module_name, name=case_name)
        df = pd.DataFrame(payload.get("records", []))
        df = self._filter_dataframe(df, filter_col=filter_col, filter_value=filter_value)
        df = self._sort_dataframe(df, sort_by=sort_by, sort_dir=sort_dir)
        total_rows = int(len(df))
        page = df.iloc[offset : offset + limit].reset_index(drop=True)
        return {
            "run_name": str(run_name),
            "module": str(module_name),
            "name": str(case_name),
            "offset": int(offset),
            "limit": int(limit),
            "total_rows": total_rows,
            "row_count": int(len(page)),
            "columns": self._column_specs(page if not page.empty else df),
            "records": as_jsonable(page.to_dict(orient="records")),
        }

    def analysis_patient_case_payload(
        self,
        *,
        run_name: str,
        module_name: str,
        patient_id: str,
        limit: int = 25,
    ) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        self._get_module_index_item(run_name=run_name, module_name=module_name)
        payload = describe_patient_case(run_root, patient_id, module_name=module_name, limit=limit)
        return {
            "run_name": str(run_name),
            "module": str(module_name),
            "patient": payload,
        }

    def comparison_payload(self, *, run_name: str) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        comparison_dir = run_root / "analysis" / "comparison"
        summary_path = comparison_dir / "summary.json"
        if not summary_path.exists():
            return {
                "run_name": str(run_name),
                "status": "missing",
                "summary": None,
                "tables": [],
                "charts": [],
                "highlights": [],
            }

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        train_df = self._read_csv_optional(comparison_dir / "train_metrics.csv")
        agent_df = self._read_csv_optional(comparison_dir / "agent_predict_metrics.csv")
        tables = []
        charts = []
        highlights = []
        if not train_df.empty:
            tables.append(self._table_payload_from_frame(name="train_metrics", df=train_df, preview_limit=8))
            charts.append(
                {
                    "id": "train_delta_mean",
                    "kind": "grouped_bar",
                    "title": "Training Metric Deltas",
                    "x_key": "model",
                    "y_key": "delta_mean",
                    "group_key": "metric",
                    "data": as_jsonable(train_df.to_dict(orient="records")),
                }
            )
            top = train_df.assign(abs_delta=train_df["delta_mean"].abs()).sort_values(
                "abs_delta", ascending=False, kind="stable"
            )
            if not top.empty:
                row = top.iloc[0]
                highlights.append(
                    {
                        "title": "Largest training delta",
                        "body": f"{row['model']} {row['metric']} delta_mean={float(row['delta_mean']):.4f}",
                    }
                )
        if not agent_df.empty:
            tables.append(self._table_payload_from_frame(name="agent_predict_metrics", df=agent_df, preview_limit=8))
            charts.append(
                {
                    "id": "agent_predict_delta_mean",
                    "kind": "grouped_bar",
                    "title": "Agent Predict Metric Deltas",
                    "x_key": "model",
                    "y_key": "delta_mean",
                    "group_key": "metric",
                    "data": as_jsonable(agent_df.to_dict(orient="records")),
                }
            )
        return {
            "run_name": str(run_name),
            "status": "ok",
            "summary": as_jsonable(summary),
            "tables": tables,
            "charts": charts,
            "highlights": highlights,
        }

    def cohort_compare_payload(
        self,
        *,
        run_name: str,
        split: str | None,
        left_role: str,
        right_role: str,
        top_k: int,
    ) -> dict[str, Any]:
        run_root = self._resolve_run(run_name)
        return {
            "run_name": str(run_name),
            "comparison": compare_cohorts(
                run_root,
                split=split,
                left_role=left_role,
                right_role=right_role,
                top_k=top_k,
            ),
        }

    def _resolve_run(self, run_name: str) -> Path:
        run_root = (self.root_dir / run_name).resolve()
        manifest = read_run_manifest(run_root)
        if manifest is None:
            raise FileNotFoundError(f"Missing run under {self.root_dir}: {run_name}")
        return run_root

    def _normalize_module_index_item(self, *, run_name: str, item: dict[str, Any]) -> dict[str, Any]:
        name = str(item.get("name"))
        meta = MODULE_META.get(name, {"title": self._titleize(name), "description": ""})
        table_paths = [str(path) for path in item.get("table_paths", []) if isinstance(path, str)]
        plot_paths = [str(path) for path in item.get("plot_paths", []) if isinstance(path, str)]
        case_paths = [str(path) for path in item.get("case_paths", []) if isinstance(path, str)]
        return {
            "name": name,
            "title": meta["title"],
            "description": meta["description"],
            "status": str(item.get("status", "unknown")),
            "summary_path": item.get("summary_path"),
            "table_paths": table_paths,
            "table_names": [Path(path).stem for path in table_paths],
            "plot_paths": plot_paths,
            "plot_names": [Path(path).stem for path in plot_paths],
            "case_paths": case_paths,
            "case_names": [Path(path).stem for path in case_paths],
            "legacy_paths": [str(path) for path in item.get("legacy_paths", []) if isinstance(path, str)],
            "details": as_jsonable(item.get("details", {})),
            "route": f"/runs/{run_name}/analysis/{name}",
        }

    def _get_module_index_item(self, *, run_name: str, module_name: str) -> dict[str, Any]:
        index = self.analysis_index_payload(run_name=run_name)
        modules = index.get("modules", [])
        for item in modules:
            if item.get("name") == module_name:
                return item
        raise FileNotFoundError(f"Missing analysis module {module_name!r} for run {run_name!r}")

    def _read_module_summary(self, *, run_root: Path, module_name: str) -> dict[str, Any]:
        try:
            return read_analysis_summary(run_root, module_name)
        except FileNotFoundError:
            return {"module": module_name, "status": "missing"}

    def _module_table_previews(self, *, run_root: Path, module_name: str, module_item: dict[str, Any]) -> list[dict[str, Any]]:
        previews = []
        for table_name in module_item.get("table_names", []):
            try:
                df = _read_analysis_table_df(run_root, module_name, table_name)
            except FileNotFoundError:
                continue
            previews.append(self._table_payload_from_frame(name=table_name, df=df, preview_limit=8))
        return previews

    def _module_charts(self, *, run_root: Path, module_name: str, module_item: dict[str, Any]) -> list[dict[str, Any]]:
        charts = []
        for plot_name in module_item.get("plot_names", []):
            try:
                spec = read_analysis_plot_spec(run_root, module_name, plot_name)
            except FileNotFoundError:
                continue
            charts.append(
                {
                    "id": str(plot_name),
                    "kind": str(spec.get("kind", "bar")),
                    "title": str(spec.get("title") or self._titleize(plot_name)),
                    "x_key": spec.get("x"),
                    "y_key": spec.get("y"),
                    "group_key": spec.get("group"),
                    "data": as_jsonable(spec.get("data", [])),
                }
            )
        if charts:
            return charts

        # Fallback charts from tables when plot specs are unavailable.
        fallback = []
        table_names = module_item.get("table_names", [])
        if module_name == "interpretability" and "artifacts" in table_names:
            df = _read_analysis_table_df(run_root, module_name, "artifacts")
            ok_rows = df[df.get("status", "").astype(str) == "ok"].copy() if "status" in df.columns else df
            if not ok_rows.empty:
                fallback.append(
                    {
                        "id": "interpretability_results",
                        "kind": "bar",
                        "title": "Interpretability Artifacts by Model",
                        "x_key": "model",
                        "y_key": "count",
                        "data": as_jsonable(
                            ok_rows.groupby("model", sort=True)
                            .size()
                            .rename("count")
                            .reset_index()
                            .to_dict(orient="records")
                        ),
                    }
                )
        return fallback

    def _summary_kpis(self, *, module_name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for field in MODULE_KPI_FIELDS.get(module_name, []):
            if field not in summary:
                continue
            out.append(self._kpi_card(field=field, value=summary.get(field)))
        if module_name == "prediction_audit" and summary.get("primary_metric") is not None:
            out.append(
                {
                    "id": "primary_metric",
                    "label": "Primary Metric",
                    "value": str(summary["primary_metric"]),
                    "format": "text",
                }
            )
        if module_name == "temporal_analysis" and summary.get("dimensions") is not None:
            out.append(
                {
                    "id": "dimensions",
                    "label": "Dimensions",
                    "value": ", ".join(str(item) for item in summary.get("dimensions", [])),
                    "format": "text",
                }
            )
        if module_name == "interpretability" and summary.get("requested_method") is not None:
            out.append(
                {
                    "id": "requested_method",
                    "label": "Requested Method",
                    "value": str(summary["requested_method"]),
                    "format": "text",
                }
            )
        for key, value in summary.items():
            if key in SUMMARY_EXCLUDE_FIELDS or key in {item["id"] for item in out}:
                continue
            if isinstance(value, (int, float, str)) and value is not None:
                out.append(self._kpi_card(field=key, value=value))
        return out

    def _module_highlights(
        self,
        *,
        run_root: Path,
        module_name: str,
        summary: dict[str, Any],
        tables: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        highlights: list[dict[str, str]] = []
        if module_name == "dataset_profile":
            top_codes = self._find_table_preview(tables, "top_codes")
            if top_codes and top_codes.get("preview"):
                first = top_codes["preview"][0]
                highlights.append(
                    {
                        "title": "Most frequent code",
                        "body": f"{first.get('code')} with count {first.get('count')}",
                    }
                )
        elif module_name == "cohort_analysis":
            try:
                comparison = compare_cohorts(run_root, split=None, left_role="train", right_role="test", top_k=3)
            except Exception:
                comparison = None
            if comparison is not None:
                gap = (comparison.get("deltas") or {}).get("label_rate_delta")
                if gap is not None:
                    highlights.append(
                        {
                            "title": "Train to test label-rate delta",
                            "body": f"{float(gap):.4f}",
                        }
                    )
        elif module_name == "prediction_audit":
            model_summary = self._find_table_preview(tables, "model_summary")
            if model_summary and model_summary.get("preview"):
                best = self._top_row_by_metric(model_summary["preview"])
                if best is not None:
                    metric_key = next((key for key in best if key.endswith("_mean")), None)
                    if metric_key is not None:
                        highlights.append(
                            {
                                "title": "Best model summary",
                                "body": f"{best.get('model')} {metric_key}={best.get(metric_key)}",
                            }
                        )
        elif module_name == "temporal_analysis":
            dimensions = summary.get("dimensions") or []
            if dimensions:
                highlights.append(
                    {
                        "title": "Segment dimensions",
                        "body": ", ".join(str(item) for item in dimensions),
                    }
                )
        elif module_name == "interpretability":
            artifacts = self._find_table_preview(tables, "artifacts")
            if artifacts and artifacts.get("preview"):
                top = next((row for row in artifacts["preview"] if row.get("status") == "ok"), None)
                if top is not None and top.get("top_feature") is not None:
                    highlights.append(
                        {
                            "title": "Top importance artifact",
                            "body": f"{top.get('model')} {top.get('method')} top feature {top.get('top_feature')}",
                        }
                    )
        elif module_name == "agent_audit":
            slices = self._find_table_preview(tables, "slices")
            if slices and slices.get("preview"):
                top = max(
                    slices["preview"],
                    key=lambda row: float(row.get("parse_success_rate", 0.0) or 0.0),
                )
                highlights.append(
                    {
                        "title": "Best parse success",
                        "body": f"{top.get('predictor_name')} {top.get('split')} rate={top.get('parse_success_rate')}",
                    }
                )
        if not highlights and summary.get("reason") is not None:
            highlights.append({"title": "Module state", "body": str(summary["reason"])})
        return highlights

    def _table_payload_from_frame(self, *, name: str, df: pd.DataFrame, preview_limit: int) -> dict[str, Any]:
        preview = df.head(preview_limit).reset_index(drop=True)
        return {
            "name": str(name),
            "title": TABLE_TITLES.get(name, self._titleize(name)),
            "row_count": int(len(df)),
            "columns": self._column_specs(df),
            "preview": as_jsonable(preview.to_dict(orient="records")),
        }

    def _find_table_preview(self, tables: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        return next((table for table in tables if table.get("name") == name), None)

    def _kpi_card(self, *, field: str, value: Any) -> dict[str, Any]:
        fmt = "text"
        if isinstance(value, bool):
            fmt = "text"
        elif isinstance(value, int):
            fmt = "integer"
        elif isinstance(value, float):
            fmt = "float"
        return {
            "id": str(field),
            "label": self._titleize(field),
            "value": value,
            "format": fmt,
        }

    def _column_specs(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return [{"name": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]

    def _filter_dataframe(
        self,
        df: pd.DataFrame,
        *,
        filter_col: str | None,
        filter_value: str | None,
    ) -> pd.DataFrame:
        if df.empty or filter_col is None or filter_value is None or filter_col not in df.columns:
            return df
        mask = df[filter_col].astype(str).str.contains(str(filter_value), case=False, na=False)
        return df.loc[mask].reset_index(drop=True)

    def _sort_dataframe(
        self,
        df: pd.DataFrame,
        *,
        sort_by: str | None,
        sort_dir: str,
    ) -> pd.DataFrame:
        if df.empty or sort_by is None or sort_by not in df.columns:
            return df
        ascending = str(sort_dir).lower() == "asc"
        return df.sort_values(sort_by, ascending=ascending, kind="stable").reset_index(drop=True)

    def _read_csv_optional(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _supports_cases(self, module_name: str) -> bool:
        return module_name in {"prediction_audit", "agent_audit"}

    def _task_label(self, task: dict[str, Any] | None) -> str:
        task = task or {}
        kind = task.get("kind")
        mode = task.get("prediction_mode")
        if kind is None and mode is None:
            return "Unknown"
        return f"{self._titleize(str(kind or 'task'))} / {self._titleize(str(mode or 'mode'))}"

    def _titleize(self, value: str) -> str:
        return str(value).replace("_", " ").strip().title()

    def _top_row_by_metric(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        metric_key = next((key for key in rows[0] if key.endswith("_mean")), None)
        if metric_key is None:
            return rows[0]
        def _metric(row: dict[str, Any]) -> float:
            raw = row.get(metric_key)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float("-inf")
        return max(rows, key=_metric)
