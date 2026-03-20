from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oneehr", description="OneEHR CLI")
    sub = parser.add_subparsers(dest="command")

    pp = sub.add_parser("preprocess", help="Bin features, generate labels, split patients")
    pp.add_argument("--config", required=True, help="Path to TOML config")

    tr = sub.add_parser("train", help="Train ML/DL models")
    tr.add_argument("--config", required=True, help="Path to TOML config")
    tr.add_argument("--force", action="store_true", help="Overwrite existing train directory")

    te = sub.add_parser("test", help="Run all systems on test set")
    te.add_argument("--config", required=True, help="Path to TOML config")
    te.add_argument("--force", action="store_true", help="Overwrite existing test directory")

    an = sub.add_parser("analyze", help="SHAP, fairness, cross-system comparison")
    an.add_argument("--config", required=True, help="Path to TOML config")
    an.add_argument("--module", default=None, help="Analysis module name (comparison, feature_importance)")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "preprocess":
        from oneehr.cli.preprocess import run_preprocess
        run_preprocess(args.config)

    elif args.command == "train":
        from oneehr.cli.train import run_train
        run_train(args.config, force=args.force)

    elif args.command == "test":
        from oneehr.cli.test import run_test
        run_test(args.config, force=args.force)

    elif args.command == "analyze":
        from oneehr.cli.analyze import run_analyze
        run_analyze(args.config, module=args.module)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
