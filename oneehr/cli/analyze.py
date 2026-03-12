"""oneehr analyze subcommand."""
from __future__ import annotations

import sys

from oneehr.analysis.reporting import (
    load_analysis_context,
    normalize_modules,
    run_analysis_suite,
)


def run_analyze(
    cfg_path: str,
    *,
    run_dir: str | None,
    method: str | None,
    modules: list[str] | None,
    compare_run: str | None,
    case_limit: int | None,
) -> None:
    from oneehr.cli._common import require_manifest, resolve_run_root
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(cfg_path)
    run_root = resolve_run_root(cfg, run_dir)
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}")

    manifest = require_manifest(run_root)
    selected_modules = normalize_modules(cfg, modules, method=method)
    ctx = load_analysis_context(cfg=cfg, run_root=run_root, manifest=manifest)
    index = run_analysis_suite(
        ctx=ctx,
        modules=selected_modules,
        method=method,
        case_limit=case_limit,
        compare_run=compare_run,
    )

    module_rows = index.get("modules", [])
    if not module_rows:
        print("No analysis modules ran.", file=sys.stderr)
        return

    for item in module_rows:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        status = item.get("status")
        summary_path = item.get("summary_path")
        print(f"[{status}] {name}: {summary_path}")
    print(f"Wrote analysis index to {(run_root / 'analysis' / 'index.json').relative_to(run_root)}")
