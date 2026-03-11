from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oneehr.analysis.feature_importance import (
    shap_importance,
    xgboost_native_importance,
)
from oneehr.artifacts.run_io import RunIO
from oneehr.data.overview_light import build_dataset_overview, build_feature_overview
from oneehr.data.splits import load_splits
from oneehr.eval.metrics import binary_metrics, regression_metrics
from oneehr.eval.tables import summarize_metrics
from oneehr.utils.io import ensure_dir, write_json


ANALYSIS_SCHEMA_VERSION = 1
PLOT_SCHEMA_VERSION = 1
SUPPORTED_MODULES = (
    "dataset_profile",
    "cohort_analysis",
    "prediction_audit",
    "temporal_analysis",
    "interpretability",
    "llm_audit",
)
SUPPORTED_FORMATS = frozenset({"json", "csv", "md", "html"})


@dataclass(frozen=True)
class AnalysisModuleResult:
    name: str
    status: str
    summary_path: str | None
    table_paths: list[str]
    plot_paths: list[str]
    case_paths: list[str]
    report_paths: list[str]
    legacy_paths: list[str]
    details: dict[str, Any]


@dataclass(frozen=True)
class AnalysisContext:
    cfg: Any
    run_root: Path
    manifest: Any
    run: RunIO
    dynamic_raw: pd.DataFrame | None
    static_raw: pd.DataFrame | None
    labels_df: pd.DataFrame | None
    dynamic_feature_columns: list[str]
    static_feature_columns: list[str] | None
    X: pd.DataFrame
    y: pd.Series
    key: pd.DataFrame | None
    splits: list[Any]
    train_summary_records: list[dict[str, Any]]
    llm_summary_records: list[dict[str, Any]]


def available_modules() -> tuple[str, ...]:
    return SUPPORTED_MODULES


def normalize_modules(cfg, modules: list[str] | None, *, method: str | None) -> list[str]:
    if modules:
        selected = [str(m) for m in modules]
    elif method is not None:
        selected = ["interpretability"]
    else:
        selected = list(cfg.analysis.default_modules)

    out: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if name not in SUPPORTED_MODULES:
            raise SystemExit(
                f"Unsupported analysis module {name!r}. "
                f"Expected one of: {', '.join(SUPPORTED_MODULES)}"
            )
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def normalize_formats(cfg, formats: list[str] | None) -> list[str]:
    selected = [str(f) for f in formats] if formats else list(cfg.analysis.formats)
    out: list[str] = []
    seen: set[str] = set()
    for fmt in selected:
        if fmt not in SUPPORTED_FORMATS:
            raise SystemExit(
                f"Unsupported analysis format {fmt!r}. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        if fmt not in seen:
            out.append(fmt)
            seen.add(fmt)
    if "json" not in seen:
        out.insert(0, "json")
    return out


def load_analysis_context(*, cfg, run_root: Path, manifest) -> AnalysisContext:
    from oneehr.data.io import load_dynamic_table_optional, load_static_table

    run = RunIO(run_root=run_root)
    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic_raw = load_dynamic_table_optional(train_dataset.dynamic)
    static_raw = load_static_table(train_dataset.static)
    labels_df = run.load_labels(manifest)
    dynamic_feature_columns = manifest.dynamic_feature_columns()
    static_all, static_feature_columns = run.load_static_all(manifest)
    del static_all

    if cfg.task.prediction_mode == "patient":
        X, y = run.load_patient_view(manifest)
        key = pd.DataFrame({"patient_id": X.index.astype(str)})
    elif cfg.task.prediction_mode == "time":
        X, y, key = run.load_time_view(manifest)
    else:
        raise SystemExit(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")

    return AnalysisContext(
        cfg=cfg,
        run_root=run_root,
        manifest=manifest,
        run=run,
        dynamic_raw=dynamic_raw,
        static_raw=static_raw,
        labels_df=labels_df,
        dynamic_feature_columns=dynamic_feature_columns,
        static_feature_columns=static_feature_columns,
        X=X,
        y=y,
        key=key,
        splits=load_splits(run_root / "splits"),
        train_summary_records=_read_summary_records(run_root / "summary.json"),
        llm_summary_records=_read_summary_records(run_root / "llm" / "summary.json"),
    )


def run_analysis_suite(
    *,
    ctx: AnalysisContext,
    modules: list[str],
    formats: list[str],
    method: str | None,
    case_limit: int | None,
    compare_run: str | None,
) -> dict[str, Any]:
    analysis_root = ensure_dir(ctx.run_root / "analysis")
    case_limit_eff = int(case_limit if case_limit is not None else ctx.cfg.analysis.case_limit)
    if case_limit_eff <= 0:
        raise SystemExit("analysis case_limit must be >= 1")
    artifacts: list[AnalysisModuleResult] = []

    module_map = {
        "dataset_profile": _module_dataset_profile,
        "cohort_analysis": _module_cohort_analysis,
        "prediction_audit": _module_prediction_audit,
        "temporal_analysis": _module_temporal_analysis,
        "interpretability": _module_interpretability,
        "llm_audit": _module_llm_audit,
    }
    for name in modules:
        result = module_map[name](
            ctx=ctx,
            analysis_root=analysis_root,
            formats=formats,
            case_limit=case_limit_eff,
            method=method,
            save_plot_specs=bool(ctx.cfg.analysis.save_plot_specs),
        )
        artifacts.append(result)

    comparison_payload = None
    if compare_run is not None:
        comparison_payload = _write_run_comparison(
            current_run_root=ctx.run_root,
            compare_run_root=Path(compare_run),
            analysis_root=analysis_root,
            formats=formats,
        )

    index = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_name": str(ctx.cfg.output.run_name),
        "run_root": str(ctx.run_root),
        "task": {
            "kind": str(ctx.cfg.task.kind),
            "prediction_mode": str(ctx.cfg.task.prediction_mode),
        },
        "modules": [
            {
                "name": item.name,
                "status": item.status,
                "summary_path": item.summary_path,
                "table_paths": item.table_paths,
                "plot_paths": item.plot_paths,
                "case_paths": item.case_paths,
                "report_paths": item.report_paths,
                "legacy_paths": item.legacy_paths,
                "details": item.details,
            }
            for item in artifacts
        ],
        "comparison": comparison_payload,
    }
    write_json(analysis_root / "index.json", index)
    if "md" in formats:
        _write_text(analysis_root / "index.md", _render_index_markdown(index))
    if "html" in formats:
        _write_text(analysis_root / "index.html", _render_html_page("OneEHR Analysis Index", _render_index_html_body(index)))
    return index


def _module_dataset_profile(
    *,
    ctx: AnalysisContext,
    analysis_root: Path,
    formats: list[str],
    case_limit: int,
    method: str | None,
    save_plot_specs: bool,
) -> AnalysisModuleResult:
    del case_limit, method
    module_dir = ensure_dir(analysis_root / "dataset_profile")
    tables: dict[str, pd.DataFrame] = {}
    plot_specs: dict[str, dict[str, Any]] = {}

    overview = None
    if ctx.dynamic_raw is not None:
        train_dataset = ctx.cfg.datasets.train if ctx.cfg.datasets is not None else ctx.cfg.dataset
        overview = build_dataset_overview(ctx.dynamic_raw, train_dataset.dynamic, top_k_codes=ctx.cfg.analysis.top_k)
        tables["top_codes"] = pd.DataFrame(overview.get("top_codes", []))
        plot_specs["top_codes"] = _bar_plot_spec(
            title="Top Dynamic Codes",
            items=overview.get("top_codes", []),
            x_key="code",
            y_key="count",
        )

    static_missingness = _build_static_missingness(ctx.static_raw)
    if static_missingness is not None:
        tables["static_missingness"] = static_missingness
        plot_specs["static_missingness"] = _bar_plot_spec(
            title="Static Missingness Rate",
            items=static_missingness.head(ctx.cfg.analysis.top_k).to_dict(orient="records"),
            x_key="column",
            y_key="missing_rate",
        )

    feature_overview = build_feature_overview(
        dynamic_feature_columns=ctx.dynamic_feature_columns,
        static_feature_columns=ctx.static_feature_columns,
    )
    feature_rows = _feature_rows(feature_overview)
    if not feature_rows.empty:
        tables["feature_space"] = feature_rows

    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "module": "dataset_profile",
        "status": "ok",
        "n_dynamic_events": 0 if ctx.dynamic_raw is None else int(len(ctx.dynamic_raw)),
        "n_patients_raw": 0
        if ctx.dynamic_raw is None
        else int(ctx.dynamic_raw["patient_id"].astype(str).nunique()),
        "n_dynamic_features": int(len(ctx.dynamic_feature_columns)),
        "n_static_features": int(len(ctx.static_feature_columns or [])),
        "n_static_columns_raw": 0 if ctx.static_raw is None else int(len(ctx.static_raw.columns) - 1),
        "split_count": int(len(ctx.splits)),
        "task_kind": str(ctx.cfg.task.kind),
        "prediction_mode": str(ctx.cfg.task.prediction_mode),
        "overview": overview,
        "feature_overview": feature_overview,
    }
    return _write_module_artifacts(
        run_root=ctx.run_root,
        module_dir=module_dir,
        module_name="dataset_profile",
        summary=summary,
        tables=tables,
        plot_specs=plot_specs,
        case_tables={},
        formats=formats,
        save_plot_specs=save_plot_specs,
        legacy_paths=[],
    )


def _module_cohort_analysis(
    *,
    ctx: AnalysisContext,
    analysis_root: Path,
    formats: list[str],
    case_limit: int,
    method: str | None,
    save_plot_specs: bool,
) -> AnalysisModuleResult:
    del case_limit, method
    module_dir = ensure_dir(analysis_root / "cohort_analysis")
    split_rows = _split_role_rows(ctx)
    drift_rows = _split_drift_rows(ctx, split_rows)
    tables = {
        "split_roles": split_rows,
        "feature_drift": drift_rows,
    }
    plots: dict[str, dict[str, Any]] = {}
    if not split_rows.empty:
        plots["role_counts"] = _grouped_bar_plot_spec(
            title="Split Role Patient Counts",
            rows=split_rows.to_dict(orient="records"),
            x_key="split",
            y_key="n_patients",
            group_key="role",
        )
        plots["label_rate"] = _grouped_bar_plot_spec(
            title="Split Role Label Rate",
            rows=split_rows.to_dict(orient="records"),
            x_key="split",
            y_key="label_rate",
            group_key="role",
        )

    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "module": "cohort_analysis",
        "status": "ok" if not split_rows.empty else "skipped",
        "split_count": int(split_rows["split"].nunique()) if not split_rows.empty else 0,
        "role_count": int(split_rows["role"].nunique()) if not split_rows.empty else 0,
        "largest_label_rate_gap": None if drift_rows.empty else float(drift_rows["abs_delta"].max()),
    }
    return _write_module_artifacts(
        run_root=ctx.run_root,
        module_dir=module_dir,
        module_name="cohort_analysis",
        summary=summary,
        tables=tables,
        plot_specs=plots,
        case_tables={},
        formats=formats,
        save_plot_specs=save_plot_specs,
        legacy_paths=[],
    )


def _module_prediction_audit(
    *,
    ctx: AnalysisContext,
    analysis_root: Path,
    formats: list[str],
    case_limit: int,
    method: str | None,
    save_plot_specs: bool,
) -> AnalysisModuleResult:
    del method
    module_dir = ensure_dir(analysis_root / "prediction_audit")
    audit_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    case_tables: dict[str, pd.DataFrame] = {}

    for rec in ctx.train_summary_records:
        model_name = str(rec.get("model"))
        split_name = str(rec.get("split"))
        pred_path = ctx.run_root / "preds" / model_name / f"{split_name}.parquet"
        if not pred_path.exists():
            continue
        preds = pd.read_parquet(pred_path).copy()
        if preds.empty:
            continue
        preds["patient_id"] = preds["patient_id"].astype(str)
        threshold = _extract_binary_threshold(rec)
        slice_rows, case_df = _prediction_slice_rows(
            ctx=ctx,
            preds=preds,
            model_name=model_name,
            split_name=split_name,
            threshold=threshold,
            case_limit=case_limit,
        )
        audit_rows.extend(slice_rows)
        if not case_df.empty:
            case_tables[f"{model_name}_{split_name}"] = case_df
        subgroup_rows.extend(
            _prediction_subgroup_rows(
                ctx=ctx,
                preds=preds,
                model_name=model_name,
                split_name=split_name,
                threshold=threshold,
            )
        )

    audit_df = pd.DataFrame(audit_rows)
    subgroup_df = pd.DataFrame(subgroup_rows)
    tables = {
        "slices": audit_df,
        "subgroups": subgroup_df,
    }
    plots: dict[str, dict[str, Any]] = {}
    if not audit_df.empty:
        metric_key = _primary_metric(ctx.cfg.task.kind)
        summary_df = _summarize_metric_frame(audit_df, group_col="model", metric_col=metric_key)
        if not summary_df.empty:
            tables["model_summary"] = summary_df
            plots["model_primary_metric"] = _bar_plot_spec(
                title=f"Primary Metric by Model ({metric_key})",
                items=summary_df.to_dict(orient="records"),
                x_key="model",
                y_key=f"{metric_key}_mean",
            )
        plots["error_rate"] = _grouped_bar_plot_spec(
            title="Prediction Error Rate by Split",
            rows=audit_df.to_dict(orient="records"),
            x_key="split",
            y_key="error_rate",
            group_key="model",
        )

    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "module": "prediction_audit",
        "status": "ok" if not audit_df.empty else "skipped",
        "n_prediction_slices": int(len(audit_df)),
        "n_subgroup_rows": int(len(subgroup_df)),
        "primary_metric": _primary_metric(ctx.cfg.task.kind),
        "stratify_by": list(ctx.cfg.analysis.stratify_by),
    }
    return _write_module_artifacts(
        run_root=ctx.run_root,
        module_dir=module_dir,
        module_name="prediction_audit",
        summary=summary,
        tables=tables,
        plot_specs=plots,
        case_tables=case_tables,
        formats=formats,
        save_plot_specs=save_plot_specs,
        legacy_paths=[],
    )


def _module_temporal_analysis(
    *,
    ctx: AnalysisContext,
    analysis_root: Path,
    formats: list[str],
    case_limit: int,
    method: str | None,
    save_plot_specs: bool,
) -> AnalysisModuleResult:
    del case_limit, method
    module_dir = ensure_dir(analysis_root / "temporal_analysis")
    rows: list[dict[str, Any]] = []
    event_counts = _patient_event_counts(ctx.dynamic_raw)

    for rec in ctx.train_summary_records:
        model_name = str(rec.get("model"))
        split_name = str(rec.get("split"))
        pred_path = ctx.run_root / "preds" / model_name / f"{split_name}.parquet"
        if not pred_path.exists():
            continue
        preds = pd.read_parquet(pred_path).copy()
        if preds.empty:
            continue
        preds["patient_id"] = preds["patient_id"].astype(str)
        preds = preds.merge(event_counts, on="patient_id", how="left")
        rows.extend(_event_count_segment_rows(ctx, preds, model_name, split_name))
        if "bin_time" in preds.columns:
            rows.extend(_bin_time_segment_rows(ctx, preds, model_name, split_name))

    seg_df = pd.DataFrame(rows)
    tables = {"segments": seg_df}
    plots: dict[str, dict[str, Any]] = {}
    if not seg_df.empty:
        plots["segment_metrics"] = _grouped_bar_plot_spec(
            title="Segment Metric by Model",
            rows=seg_df.to_dict(orient="records"),
            x_key="segment",
            y_key=_primary_metric(ctx.cfg.task.kind),
            group_key="model",
        )
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "module": "temporal_analysis",
        "status": "ok" if not seg_df.empty else "skipped",
        "n_segments": int(len(seg_df)),
        "dimensions": sorted(seg_df["dimension"].dropna().astype(str).unique().tolist()) if not seg_df.empty else [],
    }
    return _write_module_artifacts(
        run_root=ctx.run_root,
        module_dir=module_dir,
        module_name="temporal_analysis",
        summary=summary,
        tables=tables,
        plot_specs=plots,
        case_tables={},
        formats=formats,
        save_plot_specs=save_plot_specs,
        legacy_paths=[],
    )


def _module_interpretability(
    *,
    ctx: AnalysisContext,
    analysis_root: Path,
    formats: list[str],
    case_limit: int,
    method: str | None,
    save_plot_specs: bool,
) -> AnalysisModuleResult:
    del case_limit
    from oneehr.models.constants import TABULAR_MODELS
    from oneehr.models.tabular import load_tabular_model
    from oneehr.config.schema import TaskConfig

    module_dir = ensure_dir(analysis_root / "interpretability")
    tables: dict[str, pd.DataFrame] = {}
    legacy_paths: list[str] = []
    rows: list[dict[str, Any]] = []

    task_cfg = TaskConfig(kind=str(ctx.cfg.task.kind), prediction_mode=str(ctx.cfg.task.prediction_mode))
    models_dir = ctx.run_root / "models"
    if not models_dir.exists():
        summary = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "module": "interpretability",
            "status": "skipped",
            "reason": "no_models_directory",
        }
        return _write_module_artifacts(
            run_root=ctx.run_root,
            module_dir=module_dir,
            module_name="interpretability",
            summary=summary,
            tables={},
            plot_specs={},
            case_tables={},
            formats=formats,
            save_plot_specs=save_plot_specs,
            legacy_paths=[],
        )

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if model_name not in TABULAR_MODELS:
            if method == "attention":
                rows.append({"model": model_name, "split": None, "method": "attention", "status": "unsupported"})
            continue
        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name
            model_path = _resolve_tabular_model_path(split_dir)
            if model_path is None:
                continue
            art = load_tabular_model(split_dir, task=task_cfg, kind=model_name)
            methods_to_run = _interpretability_methods(model_name=model_name, method=method)
            for meth in methods_to_run:
                X_model = _analysis_input_for_feature_columns(ctx, art.feature_columns)
                if X_model is None:
                    rows.append(
                        {
                            "model": model_name,
                            "split": split_name,
                            "method": meth,
                            "status": "skipped_missing_features",
                        }
                    )
                    continue

                try:
                    if meth == "xgboost":
                        result = xgboost_native_importance(art.model, X_model, feature_names=art.feature_columns)
                    elif meth == "shap":
                        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                            result = shap_importance(
                                art.model,
                                X_model,
                                task_kind=str(ctx.cfg.task.kind),
                                feature_names=art.feature_columns,
                                nsamples=min(int(ctx.cfg.analysis.shap_max_samples), len(ctx.X)),
                            )
                    else:
                        rows.append({"model": model_name, "split": split_name, "method": meth, "status": "unsupported"})
                        continue
                except Exception as exc:  # pragma: no cover - environment-dependent deps
                    rows.append({"model": model_name, "split": split_name, "method": meth, "status": "error"})
                    rows[-1]["error_message"] = str(exc)
                    continue
                payload = {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "module": "interpretability",
                    "model": model_name,
                    "split": split_name,
                    "method": result.method,
                    "input_kind": result.input_kind,
                    "feature_names": result.feature_names,
                    "importances": result.importances.tolist(),
                }
                module_path = module_dir / f"{model_name}_{split_name}_{meth}.json"
                write_json(module_path, payload)
                legacy_path = analysis_root / f"feature_importance_{model_name}_{split_name}_{meth}.json"
                write_json(legacy_path, {
                    "method": result.method,
                    "input_kind": result.input_kind,
                    "feature_names": result.feature_names,
                    "importances": result.importances.tolist(),
                })
                legacy_paths.append(str(legacy_path.relative_to(ctx.run_root)))
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "method": meth,
                        "status": "ok",
                        "artifact_path": str(module_path.relative_to(ctx.run_root)),
                        "legacy_path": str(legacy_path.relative_to(ctx.run_root)),
                        "top_feature": result.feature_names[int(np.argmax(result.importances))] if len(result.feature_names) else None,
                    }
                )
                top_df = pd.DataFrame(
                    {
                        "feature_name": result.feature_names,
                        "importance": result.importances,
                        "model": model_name,
                        "split": split_name,
                        "method": meth,
                    }
                ).sort_values("importance", ascending=False, kind="stable")
                tables[f"{model_name}_{split_name}_{meth}"] = top_df.head(int(ctx.cfg.analysis.top_k)).reset_index(drop=True)

    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "module": "interpretability",
        "status": "ok" if rows else "skipped",
        "result_count": int(sum(1 for row in rows if row.get("status") == "ok")),
        "requested_method": method,
    }
    if rows:
        tables["artifacts"] = pd.DataFrame(rows)
    return _write_module_artifacts(
        run_root=ctx.run_root,
        module_dir=module_dir,
        module_name="interpretability",
        summary=summary,
        tables=tables,
        plot_specs={},
        case_tables={},
        formats=formats,
        save_plot_specs=save_plot_specs,
        legacy_paths=legacy_paths,
    )


def _module_llm_audit(
    *,
    ctx: AnalysisContext,
    analysis_root: Path,
    formats: list[str],
    case_limit: int,
    method: str | None,
    save_plot_specs: bool,
) -> AnalysisModuleResult:
    del method
    module_dir = ensure_dir(analysis_root / "llm_audit")
    llm_root = ctx.run_root / "llm"
    if not llm_root.exists() or not ctx.llm_summary_records:
        summary = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "module": "llm_audit",
            "status": "skipped",
            "reason": "no_llm_artifacts",
        }
        return _write_module_artifacts(
            run_root=ctx.run_root,
            module_dir=module_dir,
            module_name="llm_audit",
            summary=summary,
            tables={},
            plot_specs={},
            case_tables={},
            formats=formats,
            save_plot_specs=save_plot_specs,
            legacy_paths=[],
        )

    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    case_tables: dict[str, pd.DataFrame] = {}
    for rec in ctx.llm_summary_records:
        llm_model = str(rec.get("llm_model"))
        split_name = str(rec.get("split"))
        pred_rel = (((rec.get("artifacts") or {}).get("preds_parquet")) or None)
        if not isinstance(pred_rel, str):
            continue
        pred_path = ctx.run_root / pred_rel
        if not pred_path.exists():
            continue
        preds = pd.read_parquet(pred_path).copy()
        if preds.empty:
            continue
        parsed_rate = float(rec.get("parse_success_rate", 0.0) or 0.0)
        coverage = float(rec.get("coverage", 0.0) or 0.0)
        summary_rows.append(
            {
                "llm_model": llm_model,
                "split": split_name,
                "parse_success_rate": parsed_rate,
                "coverage": coverage,
                "avg_latency_ms": float(pd.to_numeric(preds.get("latency_ms"), errors="coerce").mean()),
                "total_prompt_tokens": int(pd.to_numeric(preds.get("token_usage_prompt"), errors="coerce").fillna(0).sum()),
                "total_completion_tokens": int(pd.to_numeric(preds.get("token_usage_completion"), errors="coerce").fillna(0).sum()),
                "total_tokens": int(pd.to_numeric(preds.get("token_usage_total"), errors="coerce").fillna(0).sum()),
                **{f"metric_{k}": v for k, v in (rec.get("metrics") or {}).items()},
            }
        )
        fail_df = preds[preds["parsed_ok"] == False].copy()  # noqa: E712
        if not fail_df.empty:
            counts = fail_df["error_code"].fillna("unknown").value_counts()
            case_tables[f"{llm_model}_{split_name}"] = (
                fail_df.sort_values("latency_ms", ascending=False, kind="stable")
                .head(case_limit)
                .reset_index(drop=True)
            )
            for code, count in counts.items():
                failure_rows.append({"llm_model": llm_model, "split": split_name, "error_code": code, "count": int(count)})

    summary_df = pd.DataFrame(summary_rows)
    failure_df = pd.DataFrame(failure_rows)
    tables = {
        "slices": summary_df,
        "failures": failure_df,
    }
    plots: dict[str, dict[str, Any]] = {}
    if not summary_df.empty:
        plots["parse_success_rate"] = _grouped_bar_plot_spec(
            title="LLM Parse Success Rate",
            rows=summary_df.to_dict(orient="records"),
            x_key="split",
            y_key="parse_success_rate",
            group_key="llm_model",
        )
        plots["token_usage"] = _grouped_bar_plot_spec(
            title="LLM Token Usage",
            rows=summary_df.to_dict(orient="records"),
            x_key="split",
            y_key="total_tokens",
            group_key="llm_model",
        )
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "module": "llm_audit",
        "status": "ok",
        "n_slices": int(len(summary_df)),
        "n_failure_buckets": int(len(failure_df)),
    }
    return _write_module_artifacts(
        run_root=ctx.run_root,
        module_dir=module_dir,
        module_name="llm_audit",
        summary=summary,
        tables=tables,
        plot_specs=plots,
        case_tables=case_tables,
        formats=formats,
        save_plot_specs=save_plot_specs,
        legacy_paths=[],
    )


def _write_module_artifacts(
    *,
    run_root: Path,
    module_dir: Path,
    module_name: str,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    plot_specs: dict[str, dict[str, Any]],
    case_tables: dict[str, pd.DataFrame],
    formats: list[str],
    save_plot_specs: bool,
    legacy_paths: list[str],
) -> AnalysisModuleResult:
    summary_path = module_dir / "summary.json"
    write_json(summary_path, summary)

    table_paths: list[str] = []
    if "csv" in formats:
        for name, df in tables.items():
            path = module_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            table_paths.append(str(path.relative_to(run_root)))

    plot_paths: list[str] = []
    if save_plot_specs and summary.get("status") != "skipped":
        plots_dir = ensure_dir(module_dir / "plots")
        for name, spec in plot_specs.items():
            path = plots_dir / f"{name}.json"
            write_json(path, spec)
            plot_paths.append(str(path.relative_to(run_root)))

    case_paths: list[str] = []
    if case_tables:
        cases_dir = ensure_dir(module_dir / "cases")
        for name, df in case_tables.items():
            path = cases_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
            case_paths.append(str(path.relative_to(run_root)))

    report_paths: list[str] = []
    if "md" in formats:
        md_path = module_dir / "summary.md"
        _write_text(md_path, _render_module_markdown(module_name, summary, tables))
        report_paths.append(str(md_path.relative_to(run_root)))
    if "html" in formats:
        html_path = module_dir / "summary.html"
        _write_text(html_path, _render_html_page(f"OneEHR {module_name}", _render_module_html_body(module_name, summary, tables)))
        report_paths.append(str(html_path.relative_to(run_root)))

    return AnalysisModuleResult(
        name=module_name,
        status=str(summary.get("status", "ok")),
        summary_path=str(summary_path.relative_to(run_root)),
        table_paths=table_paths,
        plot_paths=plot_paths,
        case_paths=case_paths,
        report_paths=report_paths,
        legacy_paths=legacy_paths,
        details={"table_count": len(table_paths), "plot_count": len(plot_paths), "case_count": len(case_paths)},
    )


def _write_run_comparison(
    *,
    current_run_root: Path,
    compare_run_root: Path,
    analysis_root: Path,
    formats: list[str],
) -> dict[str, Any]:
    current = _read_top_summary(current_run_root / "summary.json")
    other = _read_top_summary(compare_run_root / "summary.json")
    current_task = current.get("task") or {}
    other_task = other.get("task") or {}
    if current_task != other_task:
        raise SystemExit(
            "compare-run requires matching task settings. "
            f"Current={current_task!r} Compare={other_task!r}"
        )
    comparison_dir = ensure_dir(analysis_root / "comparison")
    train_delta = _compare_summary_records(
        current_records=_read_summary_records(current_run_root / "summary.json"),
        other_records=_read_summary_records(compare_run_root / "summary.json"),
        model_key="model",
    )
    if "csv" in formats:
        train_delta.to_csv(comparison_dir / "train_metrics.csv", index=False)
    llm_delta = _compare_summary_records(
        current_records=_read_summary_records(current_run_root / "llm" / "summary.json"),
        other_records=_read_summary_records(compare_run_root / "llm" / "summary.json"),
        model_key="llm_model",
    )
    if "csv" in formats and not llm_delta.empty:
        llm_delta.to_csv(comparison_dir / "llm_metrics.csv", index=False)

    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "current_run_root": str(current_run_root),
        "compare_run_root": str(compare_run_root),
        "task": current_task,
        "train_delta_rows": int(len(train_delta)),
        "llm_delta_rows": int(len(llm_delta)),
    }
    write_json(comparison_dir / "summary.json", summary)
    if "md" in formats:
        _write_text(
            comparison_dir / "summary.md",
            _render_module_markdown("comparison", summary, {"train_metrics": train_delta, "llm_metrics": llm_delta}),
        )
    if "html" in formats:
        _write_text(
            comparison_dir / "summary.html",
            _render_html_page("OneEHR comparison", _render_module_html_body("comparison", summary, {"train_metrics": train_delta, "llm_metrics": llm_delta})),
        )
    return {
        "summary_path": str((comparison_dir / "summary.json").relative_to(current_run_root)),
        "train_metrics_path": (
            None
            if train_delta.empty or "csv" not in formats
            else str((comparison_dir / "train_metrics.csv").relative_to(current_run_root))
        ),
        "llm_metrics_path": (
            None
            if llm_delta.empty or "csv" not in formats
            else str((comparison_dir / "llm_metrics.csv").relative_to(current_run_root))
        ),
    }


def _compare_summary_records(
    *,
    current_records: list[dict[str, Any]],
    other_records: list[dict[str, Any]],
    model_key: str,
) -> pd.DataFrame:
    cur = _flatten_summary_records(current_records, model_key=model_key)
    oth = _flatten_summary_records(other_records, model_key=model_key)
    if cur.empty or oth.empty:
        return pd.DataFrame()
    cur_agg = summarize_metrics(cur.rename(columns={model_key: "model"}), group_cols=["model"])
    oth_agg = summarize_metrics(oth.rename(columns={model_key: "model"}), group_cols=["model"])
    if cur_agg.empty or oth_agg.empty:
        return pd.DataFrame()
    cur_agg = cur_agg.rename(columns={"mean": "current_mean", "std": "current_std"})
    oth_agg = oth_agg.rename(columns={"mean": "compare_mean", "std": "compare_std"})
    merged = cur_agg.merge(
        oth_agg[["model", "metric", "compare_mean", "compare_std"]],
        on=["model", "metric"],
        how="inner",
    )
    if merged.empty:
        return merged
    merged["delta_mean"] = merged["current_mean"] - merged["compare_mean"]
    return merged.sort_values(["model", "metric"], kind="stable").reset_index(drop=True)


def _split_role_rows(ctx: AnalysisContext) -> pd.DataFrame:
    if not ctx.splits:
        return pd.DataFrame()
    pid_key = ctx.key if ctx.key is not None else pd.DataFrame({"patient_id": ctx.X.index.astype(str)})
    pid_key = pid_key.copy()
    pid_key["patient_id"] = pid_key["patient_id"].astype(str)
    event_counts = _patient_event_counts(ctx.dynamic_raw)
    rows: list[dict[str, Any]] = []
    for sp in ctx.splits:
        for role, patients in (
            ("train", sp.train_patients),
            ("val", sp.val_patients),
            ("test", sp.test_patients),
        ):
            patient_ids = pd.Index(np.asarray(patients, dtype=str))
            mask = pid_key["patient_id"].isin(patient_ids)
            y_role = pd.to_numeric(ctx.y.loc[mask], errors="coerce")
            label_rate = float(y_role.mean()) if y_role.notna().any() else None
            per_patient_counts = event_counts[event_counts["patient_id"].isin(patient_ids)]["n_events"]
            rows.append(
                {
                    "split": sp.name,
                    "role": role,
                    "n_patients": int(patient_ids.nunique()),
                    "n_samples": int(mask.sum()),
                    "n_labeled_samples": int(y_role.notna().sum()),
                    "label_rate": label_rate,
                    "mean_events_per_patient": float(per_patient_counts.mean()) if not per_patient_counts.empty else None,
                }
            )
    return pd.DataFrame(rows)


def _split_drift_rows(ctx: AnalysisContext, split_rows: pd.DataFrame) -> pd.DataFrame:
    if split_rows.empty:
        return pd.DataFrame()
    pid_key = ctx.key if ctx.key is not None else pd.DataFrame({"patient_id": ctx.X.index.astype(str)})
    pid_key = pid_key.copy()
    pid_key["patient_id"] = pid_key["patient_id"].astype(str)
    rows: list[dict[str, Any]] = []
    feat_cols = list(ctx.X.columns[: min(len(ctx.X.columns), int(ctx.cfg.analysis.top_k))])
    for sp in ctx.splits:
        train_mask = pid_key["patient_id"].isin(np.asarray(sp.train_patients, dtype=str))
        ref_mean = ctx.X.iloc[train_mask.to_numpy(), :][feat_cols].mean(numeric_only=True)
        for role, patients in (("val", sp.val_patients), ("test", sp.test_patients)):
            role_mask = pid_key["patient_id"].isin(np.asarray(patients, dtype=str))
            role_mean = ctx.X.iloc[role_mask.to_numpy(), :][feat_cols].mean(numeric_only=True)
            delta = (role_mean - ref_mean).abs().sort_values(ascending=False)
            for feature_name, value in delta.head(int(ctx.cfg.analysis.top_k)).items():
                rows.append(
                    {
                        "split": sp.name,
                        "role": role,
                        "feature_name": feature_name,
                        "abs_delta": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _prediction_slice_rows(
    *,
    ctx: AnalysisContext,
    preds: pd.DataFrame,
    model_name: str,
    split_name: str,
    threshold: float,
    case_limit: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    preds = preds.copy()
    preds["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")
    preds["y_pred"] = pd.to_numeric(preds["y_pred"], errors="coerce")
    preds = preds.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    if preds.empty:
        return [], pd.DataFrame()

    y_true = preds["y_true"].to_numpy(dtype=float)
    y_pred = preds["y_pred"].to_numpy(dtype=float)
    if ctx.cfg.task.kind == "binary":
        metrics = binary_metrics(y_true, y_pred).metrics
        preds["pred_label"] = (preds["y_pred"] >= float(threshold)).astype(int)
        preds["error"] = (preds["pred_label"].astype(float) != preds["y_true"].astype(float)).astype(float)
        preds["error_magnitude"] = np.abs(preds["y_pred"] - preds["y_true"])
    else:
        metrics = regression_metrics(y_true, y_pred).metrics
        preds["pred_label"] = np.nan
        preds["error"] = np.abs(preds["y_pred"] - preds["y_true"])
        preds["error_magnitude"] = preds["error"]

    row = {
        "model": model_name,
        "split": split_name,
        "n_rows": int(len(preds)),
        "threshold": float(threshold) if ctx.cfg.task.kind == "binary" else None,
        "error_rate": float(preds["error"].mean()),
        **{k: float(v) for k, v in metrics.items()},
    }
    case_df = preds.sort_values("error_magnitude", ascending=False, kind="stable").head(case_limit).reset_index(drop=True)
    return [row], case_df


def _prediction_subgroup_rows(
    *,
    ctx: AnalysisContext,
    preds: pd.DataFrame,
    model_name: str,
    split_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    if ctx.static_raw is None or not ctx.cfg.analysis.stratify_by:
        return []
    static = ctx.static_raw.copy()
    static["patient_id"] = static["patient_id"].astype(str)
    merged = preds.merge(static, on="patient_id", how="left")
    rows: list[dict[str, Any]] = []
    for col in ctx.cfg.analysis.stratify_by:
        if col not in merged.columns:
            continue
        grouped = merged.groupby(col, dropna=False, sort=True)
        for group_value, block in grouped:
            yt = pd.to_numeric(block["y_true"], errors="coerce")
            yp = pd.to_numeric(block["y_pred"], errors="coerce")
            valid = yt.notna() & yp.notna()
            if valid.sum() < 2:
                continue
            y_true = yt[valid].to_numpy(dtype=float)
            y_pred = yp[valid].to_numpy(dtype=float)
            if ctx.cfg.task.kind == "binary":
                if len(np.unique(y_true)) < 2:
                    metrics = {"accuracy": float(((y_pred >= threshold).astype(float) == y_true).mean())}
                else:
                    metrics = binary_metrics(y_true, y_pred).metrics
            else:
                metrics = regression_metrics(y_true, y_pred).metrics
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "stratify_col": col,
                    "group_value": "NA" if pd.isna(group_value) else str(group_value),
                    "n_rows": int(valid.sum()),
                    **{k: float(v) for k, v in metrics.items()},
                }
            )
    return rows


def _event_count_segment_rows(ctx: AnalysisContext, preds: pd.DataFrame, model_name: str, split_name: str) -> list[dict[str, Any]]:
    if preds.empty:
        return []
    counts = pd.to_numeric(preds["n_events"], errors="coerce").fillna(0.0)
    labels = _quantile_labels(counts, prefix="events_q")
    preds = preds.copy()
    preds["segment"] = labels
    return _segmented_metric_rows(
        preds=preds,
        task_kind=str(ctx.cfg.task.kind),
        model_name=model_name,
        split_name=split_name,
        dimension="event_count",
    )


def _bin_time_segment_rows(ctx: AnalysisContext, preds: pd.DataFrame, model_name: str, split_name: str) -> list[dict[str, Any]]:
    if "bin_time" not in preds.columns:
        return []
    tt = pd.to_datetime(preds["bin_time"], errors="coerce")
    valid = tt.notna()
    if valid.sum() == 0:
        return []
    preds = preds.loc[valid].copy()
    preds["segment"] = tt.loc[valid].dt.strftime("%Y-%m-%d")
    counts = preds["segment"].value_counts()
    keep = counts[counts >= 2].index.tolist()
    preds = preds[preds["segment"].isin(keep)].copy()
    return _segmented_metric_rows(
        preds=preds,
        task_kind=str(ctx.cfg.task.kind),
        model_name=model_name,
        split_name=split_name,
        dimension="bin_time",
    )


def _segmented_metric_rows(
    *,
    preds: pd.DataFrame,
    task_kind: str,
    model_name: str,
    split_name: str,
    dimension: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment, block in preds.groupby("segment", sort=True):
        yt = pd.to_numeric(block["y_true"], errors="coerce")
        yp = pd.to_numeric(block["y_pred"], errors="coerce")
        valid = yt.notna() & yp.notna()
        if valid.sum() < 2:
            continue
        y_true = yt[valid].to_numpy(dtype=float)
        y_pred = yp[valid].to_numpy(dtype=float)
        if task_kind == "binary":
            if len(np.unique(y_true)) < 2:
                metrics = {"accuracy": float(((y_pred >= 0.5).astype(float) == y_true).mean())}
            else:
                metrics = binary_metrics(y_true, y_pred).metrics
        else:
            metrics = regression_metrics(y_true, y_pred).metrics
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "dimension": dimension,
                "segment": segment,
                "n_rows": int(valid.sum()),
                **{k: float(v) for k, v in metrics.items()},
            }
        )
    return rows


def _build_static_missingness(static_raw: pd.DataFrame | None) -> pd.DataFrame | None:
    if static_raw is None or static_raw.empty:
        return None
    rows = []
    for col in static_raw.columns:
        if col == "patient_id":
            continue
        rows.append(
            {
                "column": col,
                "missing_count": int(static_raw[col].isna().sum()),
                "missing_rate": float(static_raw[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False, kind="stable").reset_index(drop=True)


def _feature_rows(feature_overview: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain, payload in feature_overview.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "domain": domain,
                "n_total": int(payload.get("n_total", 0)),
                "n_numeric": int(payload.get("n_numeric", 0)),
                "n_categorical": int(payload.get("n_categorical", 0)),
            }
        )
    return pd.DataFrame(rows)


def _patient_event_counts(dynamic_raw: pd.DataFrame | None) -> pd.DataFrame:
    if dynamic_raw is None or dynamic_raw.empty:
        return pd.DataFrame({"patient_id": [], "n_events": []})
    counts = dynamic_raw.groupby("patient_id", sort=True).size().rename("n_events").reset_index()
    counts["patient_id"] = counts["patient_id"].astype(str)
    return counts


def _summarize_metric_frame(df: pd.DataFrame, *, group_col: str, metric_col: str) -> pd.DataFrame:
    if df.empty or metric_col not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_col, sort=True)[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric_col}_mean", "std": f"{metric_col}_std", "count": "n"})
    )
    return grouped


def _interpretability_methods(*, model_name: str, method: str | None) -> list[str]:
    if method is not None:
        return [method]
    if model_name == "xgboost":
        return ["xgboost", "shap"]
    return ["shap"]


def _resolve_tabular_model_path(split_dir: Path) -> Path | None:
    for name in ("model.json", "model.cbm", "model.pkl"):
        path = split_dir / name
        if path.exists():
            return path
    return None


def _analysis_input_for_feature_columns(ctx: AnalysisContext, feature_columns: list[str]) -> pd.DataFrame | None:
    missing = [col for col in feature_columns if col not in ctx.X.columns]
    if missing:
        return None
    return ctx.X[feature_columns].copy()


def _extract_binary_threshold(rec: dict[str, Any]) -> float:
    metrics = rec.get("metrics") or {}
    for key in ("threshold", "val_best_threshold_cal_f1", "val_best_threshold_raw_f1"):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.5


def _primary_metric(task_kind: str) -> str:
    return "auroc" if task_kind == "binary" else "rmse"


def _quantile_labels(values: pd.Series, *, prefix: str) -> pd.Series:
    n_unique = int(values.nunique(dropna=True))
    if n_unique <= 1:
        return pd.Series([f"{prefix}_all"] * len(values), index=values.index)
    bins = min(4, n_unique)
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=[f"{prefix}{i}" for i in range(bins)], duplicates="drop").astype(str)


def _flatten_summary_records(records: list[dict[str, Any]], *, model_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        metrics = rec.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        row = {model_key: rec.get(model_key), "split": rec.get("split"), **metrics}
        for key in ("parse_success_rate", "coverage", "total_rows", "parsed_ok_rows", "ground_truth_rows", "scored_rows"):
            if key in rec:
                row[key] = rec.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def _read_summary_records(path: Path) -> list[dict[str, Any]]:
    payload = _read_top_summary(path)
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return [rec for rec in records if isinstance(rec, dict)]


def _read_top_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bar_plot_spec(*, title: str, items: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    return {
        "schema_version": PLOT_SCHEMA_VERSION,
        "kind": "bar",
        "title": title,
        "x": x_key,
        "y": y_key,
        "data": items,
    }


def _grouped_bar_plot_spec(
    *,
    title: str,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    group_key: str,
) -> dict[str, Any]:
    return {
        "schema_version": PLOT_SCHEMA_VERSION,
        "kind": "grouped_bar",
        "title": title,
        "x": x_key,
        "y": y_key,
        "group": group_key,
        "data": rows,
    }


def _render_module_markdown(module_name: str, summary: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    lines = [f"# {module_name}", "", "## Summary", ""]
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            lines.append(f"- **{key}**: `{json.dumps(value, sort_keys=True)}`")
        else:
            lines.append(f"- **{key}**: `{value}`")
    for name, df in tables.items():
        lines.extend(["", f"## {name}", ""])
        if df.empty:
            lines.append("_empty_")
        else:
            lines.append(_dataframe_markdown(df.head(20)))
    lines.append("")
    return "\n".join(lines)


def _render_module_html_body(module_name: str, summary: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    parts = [f"<h1>{escape(module_name)}</h1>", "<h2>Summary</h2>", "<ul>"]
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            rendered = escape(json.dumps(value, sort_keys=True))
        else:
            rendered = escape(str(value))
        parts.append(f"<li><strong>{escape(str(key))}</strong>: <code>{rendered}</code></li>")
    parts.append("</ul>")
    for name, df in tables.items():
        parts.append(f"<h2>{escape(name)}</h2>")
        if df.empty:
            parts.append("<p><em>empty</em></p>")
        else:
            parts.append(df.head(20).to_html(index=False, border=0))
    return "\n".join(parts)


def _render_index_markdown(index: dict[str, Any]) -> str:
    rows = []
    for item in index.get("modules", []):
        rows.append(
            {
                "module": item.get("name"),
                "status": item.get("status"),
                "summary_path": item.get("summary_path"),
                "table_count": len(item.get("table_paths") or []),
                "plot_count": len(item.get("plot_paths") or []),
                "case_count": len(item.get("case_paths") or []),
            }
        )
    lines = ["# OneEHR Analysis Index", "", f"- **run_name**: `{index.get('run_name')}`", ""]
    if rows:
        lines.append(_dataframe_markdown(pd.DataFrame(rows)))
    else:
        lines.append("_no modules_")
    if index.get("comparison") is not None:
        lines.extend(["", "## Comparison", "", f"`{json.dumps(index['comparison'], sort_keys=True)}`"])
    lines.append("")
    return "\n".join(lines)


def _render_index_html_body(index: dict[str, Any]) -> str:
    rows = []
    for item in index.get("modules", []):
        rows.append(
            {
                "module": item.get("name"),
                "status": item.get("status"),
                "summary_path": item.get("summary_path"),
                "table_count": len(item.get("table_paths") or []),
                "plot_count": len(item.get("plot_paths") or []),
                "case_count": len(item.get("case_paths") or []),
            }
        )
    parts = [f"<h1>OneEHR Analysis Index</h1>", f"<p><strong>run_name</strong>: <code>{escape(str(index.get('run_name')))}</code></p>"]
    if rows:
        parts.append(pd.DataFrame(rows).to_html(index=False, border=0))
    else:
        parts.append("<p><em>no modules</em></p>")
    if index.get("comparison") is not None:
        parts.append("<h2>Comparison</h2>")
        parts.append(f"<pre>{escape(json.dumps(index['comparison'], indent=2, sort_keys=True))}</pre>")
    return "\n".join(parts)


def _render_html_page(title: str, body: str) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            f"  <title>{escape(title)}</title>",
            "  <style>",
            "    body { font-family: Georgia, serif; margin: 2rem auto; max-width: 1100px; color: #1f2937; }",
            "    h1, h2 { color: #111827; }",
            "    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }",
            "    th, td { border: 1px solid #d1d5db; padding: 0.4rem 0.6rem; text-align: left; }",
            "    code, pre { background: #f3f4f6; padding: 0.1rem 0.3rem; }",
            "  </style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
    )


def _dataframe_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def _markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
