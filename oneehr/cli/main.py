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

    pl = sub.add_parser("plot", help="Render publication-quality figures")
    pl.add_argument("--config", required=True, help="Path to TOML config")
    pl.add_argument("--figure", nargs="*", default=None, help="Figure name(s) to render (default: all available)")
    pl.add_argument("--style", default="default", choices=["default", "nature", "lancet", "wide"], help="Journal style preset")
    pl.add_argument("--output", default=None, help="Output directory for figures")

    cv = sub.add_parser("convert", help="Convert a raw dataset into OneEHR three-table format")
    cv.add_argument("--dataset", required=True, choices=["mimic3", "mimic4", "eicu"], help="Source dataset")
    cv.add_argument("--raw-dir", required=True, help="Path to raw dataset directory")
    cv.add_argument("--output-dir", required=True, help="Output directory for converted CSVs")
    cv.add_argument("--task", default=None, help="Label task to export (default: all)")

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

    elif args.command == "plot":
        from oneehr.cli.plot import run_plot

        run_plot(args.config, figures=args.figure, style=args.style, output=args.output)

    elif args.command == "convert":
        from oneehr.cli.convert import run_convert

        run_convert(args.dataset, args.raw_dir, args.output_dir, task=args.task)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
