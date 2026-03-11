"""OneEHR CLI entry point.

Usage:
    oneehr preprocess --config <toml> [--overview] [--overview-top-k-codes N]
    oneehr train --config <toml> [--force]
    oneehr test --config <toml> [--run-dir DIR] [--test-dataset PATH] [--force] [--out-dir DIR]
    oneehr analyze --config <toml> [--run-dir DIR] [--method xgboost|shap|attention]
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
    an = sub.add_parser("analyze", help="Run feature importance analysis")
    an.add_argument("--config", required=True, help="Path to TOML config")
    an.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    an.add_argument("--method", default=None, choices=["xgboost", "shap", "attention"])

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

        run_analyze(args.config, run_dir=args.run_dir, method=args.method)

    elif args.command == "llm-preprocess":
        from oneehr.cli.llm_preprocess import run_llm_preprocess

        run_llm_preprocess(args.config, run_dir=args.run_dir, force=args.force)

    elif args.command == "llm-predict":
        from oneehr.cli.llm_predict import run_llm_predict

        run_llm_predict(args.config, run_dir=args.run_dir, force=args.force)

    else:
        parser.print_help()
        sys.exit(1)
