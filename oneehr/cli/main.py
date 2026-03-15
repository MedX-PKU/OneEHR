from __future__ import annotations

import argparse
import sys

from oneehr.cli.eval import register_eval_parser
from oneehr.cli.query import register_query_parser
from oneehr.cli.webui import register_webui_parser


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
            "Examples: dataset_profile, cohort_analysis, prediction_audit, test_audit, temporal_analysis, interpretability, agent_audit"
        ),
    )
    an.add_argument("--compare-run", default=None, help="Optional second run directory for comparison reporting")
    an.add_argument("--case-limit", type=int, default=None, help="Maximum number of case-level rows to save per analysis slice")
    an.add_argument("--method", default=None, choices=["xgboost", "shap", "attention"])

    register_eval_parser(sub)
    register_query_parser(sub)
    register_webui_parser(sub)

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
            compare_run=args.compare_run,
            case_limit=args.case_limit,
        )

    elif hasattr(args, "handler"):
        args.handler(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
