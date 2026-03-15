from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.query import (
    compare_cohorts,
    read_eval_instance,
    read_eval_trace,
)
from oneehr.utils.io import as_jsonable
from oneehr.runview import RunCatalog, RunView


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
    "test_audit": {
        "title": "Test Audit",
        "description": "External-test metrics, model ranking, and split-level evaluation deltas.",
    },
    "temporal_analysis": {
        "title": "Temporal Analysis",
        "description": "Performance segmented by event density and time windows.",
    },
    "interpretability": {
        "title": "Interpretability",
        "description": "Feature-importance artifacts across model, split, and method.",
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
    "test_audit": ["n_test_slices", "n_test_models", "best_primary_metric"],
    "temporal_analysis": ["n_segments"],
    "interpretability": ["result_count"],
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
    "metric_summary": "Metric Summary",
    "segments": "Segments",
    "artifacts": "Artifacts",
    "failures": "Failures",
    "train_metrics": "Train Metric Deltas",
    "test_metrics": "External Test Metric Deltas",
    "timeline": "Case Timeline",
    "predictions": "Case Predictions",
    "static_features": "Static Features",
    "analysis_modules": "Analysis References",
    "patient_case_matches": "Patient Case Matches",
    "leaderboard": "Evaluation Leaderboard",
    "split_metrics": "Evaluation Split Metrics",
    "pairwise": "Evaluation Pairwise Deltas",
    "trace_rows": "Evaluation Trace Rows",
}


class WebUIService:
    def __init__(self, *, root_dir: str | Path):
        self.root_dir = Path(root_dir).resolve()
        self.catalog = RunCatalog(self.root_dir)

    def list_runs_payload(self) -> dict[str, Any]:
        rows = self.catalog.list_runs()
        out = []
        for row in rows:
            item = dict(row)
            item["analysis_status"] = "ready" if item.get("has_analysis_index") else "pending"
            item["testing_status"] = "ready" if item.get("has_test_summary") else "pending"
            item["eval_status"] = "ready" if item.get("has_eval_report_summary") else "pending"
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
        run_view = self._resolve_run_view(run_name)
        run = run_view.describe()
        analysis = self.analysis_index_payload(run_name=run_name)
        hero = {
            "run_name": run["run_name"],
            "task_label": self._task_label((run.get("manifest") or {}).get("task")),
            "model_count": int(len((run.get("training") or {}).get("models") or [])),
            "test_model_count": int(len((run.get("testing") or {}).get("models") or [])),
            "analysis_module_count": int(len(analysis.get("modules", []))),
            "eval_instance_count": int(((run.get("eval") or {}).get("instance_count")) or 0),
            "eval_system_count": int(((run.get("eval") or {}).get("system_count")) or 0),
            "test_record_count": int(((run.get("testing") or {}).get("record_count")) or 0),
        }
        return {
            "run": run,
            "hero": hero,
            "analysis": analysis,
            "navigation": {
                "overview_route": f"/runs/{run_name}",
                "eval_route": f"/runs/{run_name}/eval",
                "comparison_route": f"/runs/{run_name}/comparison",
            },
        }

    def analysis_index_payload(self, *, run_name: str) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
        try:
            index = run_view.analysis_index()
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
        run_view = self._resolve_run_view(run_name)
        analysis_index = self.analysis_index_payload(run_name=run_name)
        module_item = self._get_module_index_item(run_name=run_name, module_name=module_name, index=analysis_index)
        summary = self._read_module_summary(run_view=run_view, module_name=module_name)
        tables = self._module_table_previews(run_view=run_view, module_name=module_name, module_item=module_item)
        charts = self._module_charts(run_view=run_view, module_name=module_name, module_item=module_item)
        highlights = self._module_highlights(
            run_view=run_view,
            module_name=module_name,
            summary=summary,
            tables=tables,
        )
        case_artifacts = run_view.failure_case_artifacts(module_name) if self._supports_cases(module_name) else []
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
            "comparison_available": bool(analysis_index.get("comparison")),
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
        run_view = self._resolve_run_view(run_name)
        self._get_module_index_item(run_name=run_name, module_name=module_name)
        df = run_view.analysis_table(module_name, table_name)
        df = self._filter_dataframe(df, filter_col=filter_col, filter_value=filter_value)
        df = self._sort_dataframe(df, sort_by=sort_by, sort_dir=sort_dir)
        return {
            "run_name": str(run_name),
            "module": str(module_name),
            **self._table_page_payload_from_frame(name=table_name, df=df, limit=limit, offset=offset),
        }

    def analysis_cases_index_payload(self, *, run_name: str, module_name: str) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
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
            "case_artifacts": run_view.failure_case_artifacts(module_name),
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
        run_view = self._resolve_run_view(run_name)
        self._get_module_index_item(run_name=run_name, module_name=module_name)
        cases = run_view.failure_case_artifacts(module_name)
        if case_name not in {row["name"] for row in cases}:
            raise FileNotFoundError(f"Unknown case artifact {case_name!r} for module {module_name!r}")
        payload = run_view.failure_case_rows(module_name, name=case_name)
        df = pd.DataFrame(payload.get("records", []))
        df = self._filter_dataframe(df, filter_col=filter_col, filter_value=filter_value)
        df = self._sort_dataframe(df, sort_by=sort_by, sort_dir=sort_dir)
        total_rows, page = self._paginate_dataframe(df, limit=limit, offset=offset)
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
        run_view = self._resolve_run_view(run_name)
        self._get_module_index_item(run_name=run_name, module_name=module_name)
        payload = run_view.patient_case_matches(patient_id, module_name, limit=limit)
        return {
            "run_name": str(run_name),
            "module": str(module_name),
            "patient": payload,
        }

    def eval_payload(self, *, run_name: str) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
        index = run_view.eval_index_optional()
        summary = run_view.eval_summary_optional()
        report = run_view.eval_report_summary_optional()
        if not index and not summary and not report:
            return {
                "run_name": str(run_name),
                "status": "missing",
                "index": None,
                "systems": None,
                "report": None,
                "tables": [],
                "highlights": [],
            }

        tables = []
        frames: dict[str, pd.DataFrame] = {}
        for table_name in ("leaderboard", "split_metrics", "pairwise"):
            try:
                frame = run_view.eval_table(table_name)
            except (FileNotFoundError, ValueError):
                continue
            frames[table_name] = frame
            tables.append(self._table_payload_from_frame(name=table_name, df=frame, preview_limit=8))

        highlights: list[dict[str, str]] = []
        leaderboard = frames.get("leaderboard", pd.DataFrame())
        if not leaderboard.empty:
            metric = str(report.get("primary_metric") or "accuracy")
            if metric in leaderboard.columns:
                best = leaderboard.iloc[0]
                highlights.append(
                    {
                        "title": "Top system",
                        "body": (
                            f"{best.get('system_name')} "
                            f"{metric}={self._format_highlight_value(best.get(metric))}"
                        ),
                    }
                )
        pairwise = frames.get("pairwise", pd.DataFrame())
        if not pairwise.empty:
            top = pairwise.assign(
                abs_delta=pd.to_numeric(pairwise["delta"], errors="coerce").abs()
            ).sort_values(
                "abs_delta",
                ascending=False,
                kind="stable",
            ).iloc[0]
            highlights.append(
                {
                    "title": "Largest pairwise delta",
                    "body": (
                        f"{top.get('left_system')} vs {top.get('right_system')} "
                        f"delta={self._format_highlight_value(top.get('delta'))}"
                    ),
                }
            )

        return {
            "run_name": str(run_name),
            "status": "ok",
            "index": as_jsonable(index),
            "systems": as_jsonable(summary),
            "report": as_jsonable(report),
            "tables": tables,
            "highlights": highlights,
        }

    def eval_table_payload(
        self,
        *,
        run_name: str,
        table_name: str,
        limit: int = 25,
        offset: int = 0,
        sort_by: str | None = None,
        sort_dir: str = "desc",
        filter_col: str | None = None,
        filter_value: str | None = None,
    ) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
        df = run_view.eval_table(table_name)
        df = self._filter_dataframe(df, filter_col=filter_col, filter_value=filter_value)
        df = self._sort_dataframe(df, sort_by=sort_by, sort_dir=sort_dir)
        return {
            "run_name": str(run_name),
            **self._table_page_payload_from_frame(name=table_name, df=df, limit=limit, offset=offset),
        }

    def eval_instance_payload(self, *, run_name: str, instance_id: str) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
        payload = read_eval_instance(run_view.run_root, instance_id=instance_id)
        return {
            "run_name": str(run_name),
            "instance": payload,
        }

    def eval_trace_payload(
        self,
        *,
        run_name: str,
        system_name: str,
        limit: int = 25,
        offset: int = 0,
        stage: str | None = None,
        role: str | None = None,
        round_index: int | None = None,
    ) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
        payload = read_eval_trace(
            run_view.run_root,
            system_name=system_name,
            limit=limit,
            offset=offset,
            stage=stage,
            role=role,
            round_index=round_index,
        )
        df = pd.DataFrame(payload.get("records", []))
        if df.empty and payload.get("records") == []:
            df = pd.DataFrame(columns=[])
        return {
            "run_name": str(run_name),
            "system_name": str(system_name),
            "table": "trace_rows",
            "title": TABLE_TITLES["trace_rows"],
            "offset": int(payload.get("offset", 0)),
            "limit": int(payload.get("limit", 0)),
            "total_rows": int(payload.get("total_rows", 0)),
            "row_count": int(payload.get("row_count", 0)),
            "columns": self._column_specs(df),
            "records": as_jsonable(payload.get("records", [])),
        }

    def comparison_payload(self, *, run_name: str) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
        comparison_dir = run_view.run_root / "analysis" / "comparison"
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
        comparison_tables = self._comparison_table_frames(run_view=run_view)
        train_df = comparison_tables.get("train_metrics", pd.DataFrame())
        test_df = comparison_tables.get("test_metrics", pd.DataFrame())
        tables = []
        charts = []
        highlights = []
        if "train_metrics" in comparison_tables:
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
        if "test_metrics" in comparison_tables:
            tables.append(self._table_payload_from_frame(name="test_metrics", df=test_df, preview_limit=8))
            charts.append(
                {
                    "id": "test_delta_mean",
                    "kind": "grouped_bar",
                    "title": "External Test Metric Deltas",
                    "x_key": "model",
                    "y_key": "delta_mean",
                    "group_key": "metric",
                    "data": as_jsonable(test_df.to_dict(orient="records")),
                }
            )
            top = test_df.assign(abs_delta=test_df["delta_mean"].abs()).sort_values(
                "abs_delta", ascending=False, kind="stable"
            )
            if not top.empty:
                row = top.iloc[0]
                highlights.append(
                    {
                        "title": "Largest external-test delta",
                        "body": f"{row['model']} {row['metric']} delta_mean={float(row['delta_mean']):.4f}",
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

    def comparison_table_payload(
        self,
        *,
        run_name: str,
        table_name: str,
        limit: int = 25,
        offset: int = 0,
        sort_by: str | None = None,
        sort_dir: str = "desc",
        filter_col: str | None = None,
        filter_value: str | None = None,
    ) -> dict[str, Any]:
        run_view = self._resolve_run_view(run_name)
        comparison_tables = self._comparison_table_frames(run_view=run_view)
        if table_name not in comparison_tables:
            raise FileNotFoundError(f"Missing comparison table {table_name!r} for run {run_name!r}")
        df = comparison_tables[table_name]
        df = self._filter_dataframe(df, filter_col=filter_col, filter_value=filter_value)
        df = self._sort_dataframe(df, sort_by=sort_by, sort_dir=sort_dir)
        return {
            "run_name": str(run_name),
            **self._table_page_payload_from_frame(name=table_name, df=df, limit=limit, offset=offset),
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
        run_view = self._resolve_run_view(run_name)
        return {
            "run_name": str(run_name),
            "comparison": compare_cohorts(
                run_view.run_root,
                split=split,
                left_role=left_role,
                right_role=right_role,
                top_k=top_k,
            ),
        }

    def _resolve_run_view(self, run_name: str) -> RunView:
        return self.catalog.open_run(run_name)

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

    def _get_module_index_item(
        self,
        *,
        run_name: str,
        module_name: str,
        index: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        index = index or self.analysis_index_payload(run_name=run_name)
        modules = index.get("modules", [])
        for item in modules:
            if item.get("name") == module_name:
                return item
        raise FileNotFoundError(f"Missing analysis module {module_name!r} for run {run_name!r}")

    def _read_module_summary(self, *, run_view: RunView, module_name: str) -> dict[str, Any]:
        try:
            return run_view.analysis_summary(module_name)
        except FileNotFoundError:
            return {"module": module_name, "status": "missing"}

    def _module_table_previews(
        self,
        *,
        run_view: RunView,
        module_name: str,
        module_item: dict[str, Any],
    ) -> list[dict[str, Any]]:
        previews = []
        for table_name in module_item.get("table_names", []):
            try:
                df = run_view.analysis_table(module_name, table_name)
            except FileNotFoundError:
                continue
            previews.append(self._table_payload_from_frame(name=table_name, df=df, preview_limit=8))
        return previews

    def _comparison_table_frames(self, *, run_view: RunView) -> dict[str, pd.DataFrame]:
        comparison_dir = run_view.run_root / "analysis" / "comparison"
        frames: dict[str, pd.DataFrame] = {}
        for table_name in ("train_metrics", "test_metrics"):
            df = self._read_csv_optional(comparison_dir / f"{table_name}.csv")
            if not df.empty:
                frames[table_name] = df
        return frames

    def _module_charts(
        self,
        *,
        run_view: RunView,
        module_name: str,
        module_item: dict[str, Any],
    ) -> list[dict[str, Any]]:
        charts = []
        for plot_name in module_item.get("plot_names", []):
            try:
                spec = run_view.analysis_plot_spec(module_name, plot_name)
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
            df = run_view.analysis_table(module_name, "artifacts")
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
        if module_name == "test_audit" and summary.get("primary_metric") is not None:
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
        run_view: RunView,
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
                comparison = compare_cohorts(
                    run_view.run_root,
                    split=None,
                    left_role="train",
                    right_role="test",
                    top_k=3,
                )
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
                                "body": f"{best.get('model')} {metric_key}={self._format_highlight_value(best.get(metric_key))}",
                            }
                        )
        elif module_name == "test_audit":
            model_summary = self._find_table_preview(tables, "model_summary")
            if model_summary and model_summary.get("preview"):
                best = self._top_row_by_metric(model_summary["preview"])
                if best is not None:
                    metric_key = next((key for key in best if key.endswith("_mean")), None)
                    if metric_key is not None:
                        highlights.append(
                            {
                                "title": "Best external-test model",
                                "body": f"{best.get('model')} {metric_key}={self._format_highlight_value(best.get(metric_key))}",
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
        if not highlights and summary.get("reason") is not None:
            highlights.append({"title": "Module state", "body": str(summary["reason"])})
        return highlights

    def _table_payload_from_frame(
        self,
        *,
        name: str,
        df: pd.DataFrame,
        preview_limit: int | None,
    ) -> dict[str, Any]:
        preview = df if preview_limit is None else df.head(preview_limit)
        preview = preview.reset_index(drop=True)
        preview = preview.astype(object).where(pd.notna(preview), None)
        return {
            "name": str(name),
            "title": TABLE_TITLES.get(name, self._titleize(name)),
            "row_count": int(len(df)),
            "columns": self._column_specs(df),
            "preview": as_jsonable(preview.to_dict(orient="records")),
        }

    def _table_page_payload_from_frame(
        self,
        *,
        name: str,
        df: pd.DataFrame,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        total_rows, page = self._paginate_dataframe(df, limit=limit, offset=offset)
        return {
            "table": str(name),
            "title": TABLE_TITLES.get(name, self._titleize(name)),
            "offset": int(offset),
            "limit": int(limit),
            "total_rows": total_rows,
            "row_count": int(len(page)),
            "columns": self._column_specs(page if not page.empty else df),
            "records": as_jsonable(page.to_dict(orient="records")),
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

    def _format_highlight_value(self, value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            numeric = float(value)
            if not math.isfinite(numeric):
                return "—"
            if abs(numeric) >= 1000:
                return f"{numeric:,.1f}"
            if abs(numeric) >= 1:
                return f"{numeric:.3f}".rstrip("0").rstrip(".")
            return f"{numeric:.4f}".rstrip("0").rstrip(".")
        return str(value)

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

    def _paginate_dataframe(
        self,
        df: pd.DataFrame,
        *,
        limit: int,
        offset: int,
    ) -> tuple[int, pd.DataFrame]:
        total_rows = int(len(df))
        page = df.iloc[offset : offset + limit].reset_index(drop=True)
        return total_rows, page

    def _supports_cases(self, module_name: str) -> bool:
        return module_name == "prediction_audit"

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

        metric_name = metric_key.removesuffix("_mean").lower()
        if any(token in metric_name for token in ("rmse", "mae", "mse", "loss", "error")):
            return min(rows, key=_metric)
        return max(rows, key=_metric)
