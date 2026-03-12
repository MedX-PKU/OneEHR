from __future__ import annotations

import argparse


def register_cases_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("cases", help="Build and manage durable case bundles")
    cases_sub = parser.add_subparsers(dest="cases_command")

    build = cases_sub.add_parser("build", help="Materialize case bundles under cases/")
    build.add_argument("--config", required=True, help="Path to TOML config")
    build.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    build.add_argument("--force", action="store_true", help="Overwrite existing case artifacts")
    build.set_defaults(handler=run_cases_build)


def run_cases_build(args: argparse.Namespace) -> None:
    from oneehr.cases import materialize_cases
    from oneehr.cli._common import resolve_run_root
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(args.config)
    run_root = resolve_run_root(cfg, args.run_dir)
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}. Run `oneehr preprocess` first.")
    materialize_cases(cfg, run_root=run_root, force=bool(args.force))
