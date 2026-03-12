from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oneehr.utils.io import as_jsonable


def register_query_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("query", help="Read run artifacts as JSON for notebooks, agents, and web UIs")
    query_sub = parser.add_subparsers(dest="query_group")

    _register_runs_parser(query_sub)
    _register_prompts_parser(query_sub)
    _register_cases_parser(query_sub)
    _register_analysis_parser(query_sub)
    _register_cohorts_parser(query_sub)
    _register_agent_parser(query_sub)


def _register_runs_parser(query_sub) -> None:
    parser = query_sub.add_parser("runs", help="Query run directories")
    sub = parser.add_subparsers(dest="query_action")

    list_cmd = sub.add_parser("list", help="List runs under a root directory")
    list_cmd.add_argument("--root", default="logs", help="Root directory containing runs")
    list_cmd.set_defaults(handler=_run_runs_list)

    describe_cmd = sub.add_parser("describe", help="Describe a single run")
    _add_run_locator_args(describe_cmd)
    describe_cmd.set_defaults(handler=_run_runs_describe)


def _register_prompts_parser(query_sub) -> None:
    parser = query_sub.add_parser("prompts", help="Query prompt templates")
    sub = parser.add_subparsers(dest="query_action")

    list_cmd = sub.add_parser("list", help="List prompt templates")
    list_cmd.add_argument("--family", default=None, help="Optional family filter")
    list_cmd.set_defaults(handler=_run_prompts_list)

    describe_cmd = sub.add_parser("describe", help="Describe one prompt template")
    describe_cmd.add_argument("--template", required=True, help="Prompt template name")
    describe_cmd.set_defaults(handler=_run_prompts_describe)


def _register_cases_parser(query_sub) -> None:
    parser = query_sub.add_parser("cases", help="Query durable case bundles")
    sub = parser.add_subparsers(dest="query_action")

    index_cmd = sub.add_parser("index", help="Read the cases index")
    _add_run_locator_args(index_cmd)
    index_cmd.set_defaults(handler=_run_cases_index)

    list_cmd = sub.add_parser("list", help="List case summaries")
    _add_run_locator_args(list_cmd)
    list_cmd.add_argument("--limit", type=int, default=None, help="Optional max rows to return")
    list_cmd.set_defaults(handler=_run_cases_list)

    read_cmd = sub.add_parser("read", help="Read one case bundle")
    _add_run_locator_args(read_cmd)
    _add_case_args(read_cmd)
    read_cmd.set_defaults(handler=_run_cases_read)

    timeline_cmd = sub.add_parser("timeline", help="Read the timeline for one case")
    _add_run_locator_args(timeline_cmd)
    _add_case_args(timeline_cmd)
    timeline_cmd.set_defaults(handler=_run_cases_timeline)

    static_cmd = sub.add_parser("static", help="Read static features for one case")
    _add_run_locator_args(static_cmd)
    static_cmd.add_argument("--case-id", required=True, help="Case identifier")
    static_cmd.set_defaults(handler=_run_cases_static)

    predictions_cmd = sub.add_parser("predictions", help="Read predictions for one case")
    _add_run_locator_args(predictions_cmd)
    _add_case_args(predictions_cmd)
    predictions_cmd.add_argument("--origin", default=None, help="Optional origin filter: model or agent")
    predictions_cmd.add_argument("--predictor-name", default=None, help="Optional concrete predictor/backend name")
    predictions_cmd.set_defaults(handler=_run_cases_predictions)

    evidence_cmd = sub.add_parser("evidence", help="Collect case evidence in one JSON payload")
    _add_run_locator_args(evidence_cmd)
    _add_case_args(evidence_cmd)
    evidence_cmd.set_defaults(handler=_run_cases_evidence)

    prompt_cmd = sub.add_parser("render-prompt", help="Render a prediction or review prompt for one case")
    _add_run_locator_args(prompt_cmd)
    _add_case_args(prompt_cmd)
    prompt_cmd.add_argument("--template", default=None, help="Optional prompt template name")
    prompt_cmd.add_argument("--origin", default=None, help="Optional target prediction origin for review prompts")
    prompt_cmd.add_argument("--predictor-name", default=None, help="Optional target predictor/backend name")
    prompt_cmd.set_defaults(handler=_run_cases_render_prompt)


def _register_analysis_parser(query_sub) -> None:
    parser = query_sub.add_parser("analysis", help="Query analysis artifacts")
    sub = parser.add_subparsers(dest="query_action")

    modules_cmd = sub.add_parser("modules", help="List available analysis modules")
    _add_run_locator_args(modules_cmd)
    modules_cmd.set_defaults(handler=_run_analysis_modules)

    index_cmd = sub.add_parser("index", help="Read analysis index")
    _add_run_locator_args(index_cmd)
    index_cmd.set_defaults(handler=_run_analysis_index)

    summary_cmd = sub.add_parser("summary", help="Read one analysis summary")
    _add_run_locator_args(summary_cmd)
    summary_cmd.add_argument("--module", required=True, help="Analysis module name")
    summary_cmd.set_defaults(handler=_run_analysis_summary)

    table_cmd = sub.add_parser("table", help="Read one analysis table")
    _add_run_locator_args(table_cmd)
    table_cmd.add_argument("--module", required=True, help="Analysis module name")
    table_cmd.add_argument("--table", required=True, help="Analysis table name")
    table_cmd.add_argument("--limit", type=int, default=None, help="Optional max rows to return")
    table_cmd.set_defaults(handler=_run_analysis_table)

    plot_cmd = sub.add_parser("plot", help="Read one analysis plot specification")
    _add_run_locator_args(plot_cmd)
    plot_cmd.add_argument("--module", required=True, help="Analysis module name")
    plot_cmd.add_argument("--plot", required=True, help="Plot name")
    plot_cmd.set_defaults(handler=_run_analysis_plot)

    failures_cmd = sub.add_parser("failures", help="List failure-case artifacts for one analysis module")
    _add_run_locator_args(failures_cmd)
    failures_cmd.add_argument("--module", default="prediction_audit", help="Analysis module name")
    failures_cmd.set_defaults(handler=_run_analysis_failures)

    failure_cases_cmd = sub.add_parser("failure-cases", help="Read failure-case rows for one module")
    _add_run_locator_args(failure_cases_cmd)
    failure_cases_cmd.add_argument("--module", default="prediction_audit", help="Analysis module name")
    failure_cases_cmd.add_argument("--name", default=None, help="Optional failure artifact name")
    failure_cases_cmd.add_argument("--limit", type=int, default=None, help="Optional max rows to return")
    failure_cases_cmd.set_defaults(handler=_run_analysis_failure_cases)

    patient_case_cmd = sub.add_parser("patient-case", help="Describe failure-case matches for one patient")
    _add_run_locator_args(patient_case_cmd)
    patient_case_cmd.add_argument("--module", default="prediction_audit", help="Analysis module name")
    patient_case_cmd.add_argument("--patient-id", required=True, help="Patient identifier")
    patient_case_cmd.add_argument("--limit", type=int, default=None, help="Optional max matches to return")
    patient_case_cmd.set_defaults(handler=_run_analysis_patient_case)


def _register_cohorts_parser(query_sub) -> None:
    parser = query_sub.add_parser("cohorts", help="Query cohort-level analysis")
    sub = parser.add_subparsers(dest="query_action")

    compare_cmd = sub.add_parser("compare", help="Compare two cohort roles within one split")
    _add_run_locator_args(compare_cmd)
    compare_cmd.add_argument("--split", default=None, help="Split name")
    compare_cmd.add_argument("--left-role", default="train", choices=["train", "val", "test"])
    compare_cmd.add_argument("--right-role", default="test", choices=["train", "val", "test"])
    compare_cmd.add_argument("--top-k", type=int, default=10, help="Top-k feature drift rows")
    compare_cmd.set_defaults(handler=_run_cohorts_compare)


def _register_agent_parser(query_sub) -> None:
    parser = query_sub.add_parser("agent", help="Query agent prediction and review summaries")
    sub = parser.add_subparsers(dest="query_action")

    predict_cmd = sub.add_parser("predict-summary", help="Read agent prediction summary")
    _add_run_locator_args(predict_cmd)
    predict_cmd.set_defaults(handler=_run_agent_predict_summary)

    review_cmd = sub.add_parser("review-summary", help="Read agent review summary")
    _add_run_locator_args(review_cmd)
    review_cmd.set_defaults(handler=_run_agent_review_summary)


def _run_runs_list(args: argparse.Namespace) -> None:
    from oneehr.query import list_runs

    _emit({"query": "runs.list", "root": str(Path(args.root)), "runs": list_runs(args.root)})


def _run_runs_describe(args: argparse.Namespace) -> None:
    from oneehr.query import describe_run

    run_root = _resolve_run_root(args, label="query runs describe")
    _emit({"query": "runs.describe", "run": describe_run(run_root)})


def _run_prompts_list(args: argparse.Namespace) -> None:
    from oneehr.query import list_prompt_templates

    _emit(
        {
            "query": "prompts.list",
            "family": None if args.family is None else str(args.family),
            "templates": list_prompt_templates(family=args.family),
        }
    )


def _run_prompts_describe(args: argparse.Namespace) -> None:
    from oneehr.query import describe_prompt_template

    _emit({"query": "prompts.describe", "template": describe_prompt_template(args.template)})


def _run_cases_index(args: argparse.Namespace) -> None:
    from oneehr.query import read_cases_index

    run_root = _resolve_run_root(args, label="query cases index")
    _emit({"query": "cases.index", "run_dir": str(run_root), "index": read_cases_index(run_root)})


def _run_cases_list(args: argparse.Namespace) -> None:
    from oneehr.query import list_cases

    run_root = _resolve_run_root(args, label="query cases list")
    _emit({"query": "cases.list", "run_dir": str(run_root), "cases": list_cases(run_root, limit=args.limit)})


def _run_cases_read(args: argparse.Namespace) -> None:
    from oneehr.query import read_case

    run_root = _resolve_run_root(args, label="query cases read")
    _emit(
        {
            "query": "cases.read",
            "run_dir": str(run_root),
            "case": read_case(run_root, args.case_id, limit=args.limit),
        }
    )


def _run_cases_timeline(args: argparse.Namespace) -> None:
    from oneehr.query import get_case_timeline

    run_root = _resolve_run_root(args, label="query cases timeline")
    _emit(
        {
            "query": "cases.timeline",
            "run_dir": str(run_root),
            "timeline": get_case_timeline(run_root, args.case_id, limit=args.limit),
        }
    )


def _run_cases_static(args: argparse.Namespace) -> None:
    from oneehr.query import get_case_static

    run_root = _resolve_run_root(args, label="query cases static")
    _emit({"query": "cases.static", "run_dir": str(run_root), "static": get_case_static(run_root, args.case_id)})


def _run_cases_predictions(args: argparse.Namespace) -> None:
    from oneehr.query import get_case_predictions

    run_root = _resolve_run_root(args, label="query cases predictions")
    _emit(
        {
            "query": "cases.predictions",
            "run_dir": str(run_root),
            "predictions": get_case_predictions(
                run_root,
                args.case_id,
                origin=args.origin,
                predictor_name=args.predictor_name,
                limit=args.limit,
            ),
        }
    )


def _run_cases_evidence(args: argparse.Namespace) -> None:
    from oneehr.query import collect_case_evidence

    run_root = _resolve_run_root(args, label="query cases evidence")
    _emit(
        {
            "query": "cases.evidence",
            "run_dir": str(run_root),
            "evidence": collect_case_evidence(run_root, args.case_id, limit=args.limit),
        }
    )


def _run_cases_render_prompt(args: argparse.Namespace) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.query import render_case_prompt

    if args.config is None:
        raise SystemExit("query cases render-prompt requires --config so prompt settings can be resolved.")
    cfg = load_experiment_config(args.config)
    run_root = _resolve_run_root(args, label="query cases render-prompt")
    _emit(
        {
            "query": "cases.render_prompt",
            "run_dir": str(run_root),
            "prompt": render_case_prompt(
                cfg=cfg,
                run_root=run_root,
                case_id=args.case_id,
                template_name=args.template,
                origin=args.origin,
                predictor_name=args.predictor_name,
            ),
        }
    )


def _run_analysis_modules(args: argparse.Namespace) -> None:
    from oneehr.query import list_analysis_modules

    run_root = _resolve_run_root(args, label="query analysis modules")
    _emit({"query": "analysis.modules", "run_dir": str(run_root), "modules": list_analysis_modules(run_root)})


def _run_analysis_index(args: argparse.Namespace) -> None:
    from oneehr.query import read_analysis_index

    run_root = _resolve_run_root(args, label="query analysis index")
    _emit({"query": "analysis.index", "run_dir": str(run_root), "index": read_analysis_index(run_root)})


def _run_analysis_summary(args: argparse.Namespace) -> None:
    from oneehr.query import read_analysis_summary

    run_root = _resolve_run_root(args, label="query analysis summary")
    _emit(
        {
            "query": "analysis.summary",
            "run_dir": str(run_root),
            "summary": read_analysis_summary(run_root, args.module),
        }
    )


def _run_analysis_table(args: argparse.Namespace) -> None:
    from oneehr.query import read_analysis_table

    run_root = _resolve_run_root(args, label="query analysis table")
    _emit(
        {
            "query": "analysis.table",
            "run_dir": str(run_root),
            "table": read_analysis_table(run_root, args.module, args.table, limit=args.limit),
        }
    )


def _run_analysis_plot(args: argparse.Namespace) -> None:
    from oneehr.query import read_analysis_plot_spec

    run_root = _resolve_run_root(args, label="query analysis plot")
    _emit(
        {
            "query": "analysis.plot",
            "run_dir": str(run_root),
            "plot": read_analysis_plot_spec(run_root, args.module, args.plot),
        }
    )


def _run_analysis_failures(args: argparse.Namespace) -> None:
    from oneehr.query import list_failure_cases

    run_root = _resolve_run_root(args, label="query analysis failures")
    _emit(
        {
            "query": "analysis.failures",
            "run_dir": str(run_root),
            "cases": list_failure_cases(run_root, module_name=args.module),
        }
    )


def _run_analysis_failure_cases(args: argparse.Namespace) -> None:
    from oneehr.query import read_failure_cases

    run_root = _resolve_run_root(args, label="query analysis failure-cases")
    _emit(
        {
            "query": "analysis.failure_cases",
            "run_dir": str(run_root),
            "cases": read_failure_cases(run_root, module_name=args.module, name=args.name, limit=args.limit),
        }
    )


def _run_analysis_patient_case(args: argparse.Namespace) -> None:
    from oneehr.query import describe_patient_case

    run_root = _resolve_run_root(args, label="query analysis patient-case")
    _emit(
        {
            "query": "analysis.patient_case",
            "run_dir": str(run_root),
            "patient": describe_patient_case(
                run_root,
                args.patient_id,
                module_name=args.module,
                limit=args.limit,
            ),
        }
    )


def _run_cohorts_compare(args: argparse.Namespace) -> None:
    from oneehr.query import compare_cohorts

    run_root = _resolve_run_root(args, label="query cohorts compare")
    _emit(
        {
            "query": "cohorts.compare",
            "run_dir": str(run_root),
            "comparison": compare_cohorts(
                run_root,
                split=args.split,
                left_role=args.left_role,
                right_role=args.right_role,
                top_k=args.top_k,
            ),
        }
    )


def _run_agent_predict_summary(args: argparse.Namespace) -> None:
    from oneehr.query import read_agent_predict_summary

    run_root = _resolve_run_root(args, label="query agent predict-summary")
    _emit(
        {
            "query": "agent.predict_summary",
            "run_dir": str(run_root),
            "summary": read_agent_predict_summary(run_root),
        }
    )


def _run_agent_review_summary(args: argparse.Namespace) -> None:
    from oneehr.query import read_agent_review_summary

    run_root = _resolve_run_root(args, label="query agent review-summary")
    _emit(
        {
            "query": "agent.review_summary",
            "run_dir": str(run_root),
            "summary": read_agent_review_summary(run_root),
        }
    )


def _add_run_locator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Optional config used to resolve the run directory")
    parser.add_argument("--run-dir", default=None, help="Run directory for run-scoped queries")


def _add_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-id", required=True, help="Case identifier")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to return")


def _resolve_run_root(args: argparse.Namespace, *, label: str) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir)
    if args.config is None:
        raise SystemExit(f"{label} requires --run-dir or --config")
    from oneehr.cli._common import resolve_run_root
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(args.config)
    return resolve_run_root(cfg, None)


def _emit(payload: object) -> None:
    sys.stdout.write(json.dumps(as_jsonable(payload), indent=2, sort_keys=True) + "\n")
