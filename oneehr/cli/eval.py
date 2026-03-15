from __future__ import annotations

import argparse
import json
from pathlib import Path

from oneehr.cli._common import resolve_run_root


def register_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="Build, run, and inspect unified evaluation workflows")
    eval_sub = parser.add_subparsers(dest="eval_command")

    build = eval_sub.add_parser("build", help="Materialize frozen eval instances and evidence bundles")
    build.add_argument("--config", required=True, help="Path to TOML config")
    build.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    build.add_argument("--force", action="store_true", help="Overwrite existing eval artifacts")
    build.set_defaults(handler=run_eval_build)

    run = eval_sub.add_parser("run", help="Execute configured predictor systems over frozen eval instances")
    run.add_argument("--config", required=True, help="Path to TOML config")
    run.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    run.add_argument("--force", action="store_true", help="Overwrite existing system outputs")
    run.set_defaults(handler=run_eval_run)

    report = eval_sub.add_parser("report", help="Compute leaderboard, system metrics, and paired comparisons")
    report.add_argument("--config", required=True, help="Path to TOML config")
    report.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    report.add_argument("--force", action="store_true", help="Overwrite existing report artifacts")
    report.set_defaults(handler=run_eval_report)

    trace = eval_sub.add_parser("trace", help="Read saved framework trace rows")
    trace.add_argument("--config", required=True, help="Path to TOML config")
    trace.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    trace.add_argument("--system", required=True, help="System name")
    trace.add_argument("--limit", type=int, default=25)
    trace.add_argument("--offset", type=int, default=0)
    trace.add_argument("--stage", default=None)
    trace.add_argument("--role", default=None)
    trace.add_argument("--round", dest="round_index", type=int, default=None)
    trace.set_defaults(handler=run_eval_trace)

    instance = eval_sub.add_parser("instance", help="Read one eval instance and aligned system outputs")
    instance.add_argument("--config", required=True, help="Path to TOML config")
    instance.add_argument("--run-dir", default=None, help="Run directory (overrides config)")
    instance.add_argument("--instance-id", required=True, help="Instance identifier")
    instance.set_defaults(handler=run_eval_instance)


def _load_cfg_and_root(config_path: str, run_dir: str | None):
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(config_path)
    run_root = resolve_run_root(cfg, run_dir)
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}. Run `oneehr preprocess` first.")
    return cfg, run_root


def run_eval_build(args: argparse.Namespace) -> None:
    from oneehr.eval.workflow import build_eval_artifacts

    cfg, run_root = _load_cfg_and_root(args.config, args.run_dir)
    result = build_eval_artifacts(cfg, run_root=run_root, force=bool(args.force))
    print(
        json.dumps(
            {
                "command": "eval.build",
                "run_dir": str(run_root),
                "instance_count": int(result.instance_count),
                "index_path": str(result.index_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_eval_run(args: argparse.Namespace) -> None:
    from oneehr.eval.workflow import run_eval_systems

    cfg, run_root = _load_cfg_and_root(args.config, args.run_dir)
    result = run_eval_systems(cfg, run_root=run_root, force=bool(args.force))
    print(
        json.dumps(
            {
                "command": "eval.run",
                "run_dir": str(run_root),
                "system_count": int(result.system_count),
                "summary_path": str(result.summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_eval_report(args: argparse.Namespace) -> None:
    from oneehr.eval.workflow import build_eval_report

    cfg, run_root = _load_cfg_and_root(args.config, args.run_dir)
    result = build_eval_report(cfg, run_root=run_root, force=bool(args.force))
    print(
        json.dumps(
            {
                "command": "eval.report",
                "run_dir": str(run_root),
                "leaderboard_rows": int(result.leaderboard_rows),
                "summary_path": str(result.summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_eval_trace(args: argparse.Namespace) -> None:
    from oneehr.eval.query import read_trace_rows

    _, run_root = _load_cfg_and_root(args.config, args.run_dir)
    payload = read_trace_rows(
        run_root,
        system_name=args.system,
        limit=int(args.limit),
        offset=int(args.offset),
        stage=args.stage,
        role=args.role,
        round_index=args.round_index,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_eval_instance(args: argparse.Namespace) -> None:
    from oneehr.eval.query import read_instance_payload

    _, run_root = _load_cfg_and_root(args.config, args.run_dir)
    payload = read_instance_payload(run_root, instance_id=args.instance_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
