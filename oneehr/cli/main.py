"""OneEHR CLI entry point.

Usage:
    oneehr preprocess --config <toml> [--overview] [--overview-top-k-codes N]
    oneehr train --config <toml> [--force]
    oneehr test --config <toml> [--run-dir DIR] [--test-dataset PATH] [--force] [--out-dir DIR]
    oneehr analyze --config <toml> [--run-dir DIR] [--module NAME] [--format FMT] [--compare-run DIR] [--case-limit N] [--method xgboost|shap|attention]
    oneehr inspect --tool TOOL [--config <toml> | --run-dir DIR | --root DIR] [--module NAME] [--table NAME] [--plot NAME] [--patient-id ID]
    oneehr llm-preprocess --config <toml> [--run-dir DIR] [--force]
    oneehr llm-predict --config <toml> [--run-dir DIR] [--force]
"""
from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oneehr", description="OneEHR CLI")
    sub = parser.add_subparsers(dest="command")

    # preprocess
    pp = sub.add_parser("preprocess", help="Run data preprocessing pipeline")
    pp.add_argument("--config", required=True, help="Path to TOML config")
    pp.add_argument("--overview", action="store_true", help="Print dataset overview JSON")
    pp.add_argument("--overview-top-k-codes", type=int, default=20)

    # train
    tr = sub.add_parser("train", help="Train models (includes HPO if configured)")
    tr.add_argument("--config", required=True, help="Path to TOML config")
    tr.add_argument("--force", action="store_true", help="Overwrite existing run directory")

    # test
    te = sub.add_parser("test", help="Evaluate trained models on test data")
    te.add_argument("--config", required=True, help="Path to TOML config")
    te.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    te.add_argument("--test-dataset", default=None, help="Path to test dataset config")
    te.add_argument("--force", action="store_true")
    te.add_argument("--out-dir", default=None, help="Output directory for test results")

    # analyze
    an = sub.add_parser("analyze", help="Run modular analysis, reports, and interpretability")
    an.add_argument("--config", required=True, help="Path to TOML config")
    an.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    an.add_argument(
        "--module",
        action="append",
        default=None,
        help=(
            "Analysis module to run. Repeatable. "
            "Examples: dataset_profile, cohort_analysis, prediction_audit, temporal_analysis, interpretability, llm_audit"
        ),
    )
    an.add_argument(
        "--format",
        action="append",
        default=None,
        help="Output format to write. Repeatable. Choices are json, csv, md, html.",
    )
    an.add_argument("--compare-run", default=None, help="Optional second run directory for comparison reporting")
    an.add_argument("--case-limit", type=int, default=None, help="Maximum number of case-level rows to save per analysis slice")
    an.add_argument("--method", default=None, choices=["xgboost", "shap", "attention"])

    # inspect
    ins = sub.add_parser("inspect", help="Read run and analysis artifacts as JSON for agents")
    ins.add_argument("--tool", required=True, help="Tool name, e.g. runs.list, analysis.read_summary, cases.describe_patient")
    ins.add_argument("--config", default=None, help="Optional config used to resolve the run directory")
    ins.add_argument("--run-dir", default=None, help="Run directory for run-scoped tools")
    ins.add_argument("--root", default="logs", help="Root directory for runs.list")
    ins.add_argument("--module", default=None, help="Analysis or case module name")
    ins.add_argument("--table", default=None, help="Analysis table name for analysis.read_table")
    ins.add_argument("--plot", default=None, help="Plot name for analysis.read_plot_spec")
    ins.add_argument("--patient-id", default=None, help="Patient identifier for cases.describe_patient")
    ins.add_argument("--name", default=None, help="Optional failure case artifact name")
    ins.add_argument("--split", default=None, help="Split name for cohorts.compare")
    ins.add_argument("--left-role", default="train", choices=["train", "val", "test"])
    ins.add_argument("--right-role", default="test", choices=["train", "val", "test"])
    ins.add_argument("--limit", type=int, default=None, help="Optional max rows to return for table/case queries")
    ins.add_argument("--top-k", type=int, default=10, help="Top-k feature drift rows for cohorts.compare")

    # llm-preprocess
    lp = sub.add_parser("llm-preprocess", help="Materialize LLM prompt instances from EHR artifacts")
    lp.add_argument("--config", required=True, help="Path to TOML config")
    lp.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    lp.add_argument("--force", action="store_true", help="Overwrite existing LLM instance artifacts")

    # llm-predict
    lpr = sub.add_parser("llm-predict", help="Run OpenAI-compatible LLM prediction/evaluation")
    lpr.add_argument("--config", required=True, help="Path to TOML config")
    lpr.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    lpr.add_argument("--force", action="store_true", help="Overwrite existing LLM prediction artifacts")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "preprocess":
        from oneehr.cli.preprocess import run_preprocess

        run_preprocess(args.config, overview=args.overview, overview_top_k_codes=args.overview_top_k_codes)

    elif args.command == "train":
        from oneehr.cli.train import run_train

        run_train(args.config, force=args.force)

    elif args.command == "test":
        from oneehr.cli.test import run_test

        run_test(
            args.config,
            run_dir=args.run_dir,
            test_dataset=args.test_dataset,
            force=args.force,
            out_dir=args.out_dir,
        )

    elif args.command == "analyze":
        from oneehr.cli.analyze import run_analyze

        run_analyze(
            args.config,
            run_dir=args.run_dir,
            method=args.method,
            modules=args.module,
            formats=args.format,
            compare_run=args.compare_run,
            case_limit=args.case_limit,
        )

    elif args.command == "inspect":
        from oneehr.cli.inspect import run_inspect

        run_inspect(
            tool=args.tool,
            config=args.config,
            run_dir=args.run_dir,
            root=args.root,
            module=args.module,
            table=args.table,
            plot=args.plot,
            patient_id=args.patient_id,
            name=args.name,
            split=args.split,
            left_role=args.left_role,
            right_role=args.right_role,
            limit=args.limit,
            top_k=args.top_k,
        )

    elif args.command == "llm-preprocess":
        from oneehr.cli.llm_preprocess import run_llm_preprocess

        run_llm_preprocess(args.config, run_dir=args.run_dir, force=args.force)

    elif args.command == "llm-predict":
        from oneehr.cli.llm_predict import run_llm_predict

        run_llm_predict(args.config, run_dir=args.run_dir, force=args.force)

    else:
        parser.print_help()
        sys.exit(1)
