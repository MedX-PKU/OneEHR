"""oneehr inspect subcommand."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from oneehr.utils.io import as_jsonable


SUPPORTED_TOOLS = (
    "runs.list",
    "runs.describe",
    "analysis.list_modules",
    "analysis.read_index",
    "analysis.read_summary",
    "analysis.read_table",
    "analysis.read_plot_spec",
    "cases.list_failures",
    "cases.read_failures",
    "cases.describe_patient",
    "cohorts.compare",
)


def run_inspect(
    *,
    tool: str,
    config: str | None,
    run_dir: str | None,
    root: str | None,
    module: str | None,
    table: str | None,
    plot: str | None,
    patient_id: str | None,
    name: str | None,
    split: str | None,
    left_role: str,
    right_role: str,
    limit: int | None,
    top_k: int,
) -> None:
    from oneehr.agent import (
        compare_cohorts,
        describe_patient_case,
        describe_run,
        list_analysis_modules,
        list_failure_cases,
        list_runs,
        read_analysis_index,
        read_analysis_plot_spec,
        read_analysis_summary,
        read_analysis_table,
        read_failure_cases,
    )

    if tool not in SUPPORTED_TOOLS:
        raise SystemExit(f"Unsupported inspect tool {tool!r}. Expected one of: {', '.join(SUPPORTED_TOOLS)}")

    run_root = _resolve_run_root(config=config, run_dir=run_dir, tool=tool)
    payload: object

    if tool == "runs.list":
        payload = {
            "tool": tool,
            "root": str(Path(root or "logs")),
            "runs": list_runs(root or "logs"),
        }
    elif tool == "runs.describe":
        payload = {"tool": tool, "run": describe_run(run_root)}
    elif tool == "analysis.list_modules":
        payload = {"tool": tool, "run_dir": str(run_root), "modules": list_analysis_modules(run_root)}
    elif tool == "analysis.read_index":
        payload = {"tool": tool, "run_dir": str(run_root), "index": read_analysis_index(run_root)}
    elif tool == "analysis.read_summary":
        module_name = _require_arg("module", module)
        payload = {"tool": tool, "run_dir": str(run_root), "summary": read_analysis_summary(run_root, module_name)}
    elif tool == "analysis.read_table":
        module_name = _require_arg("module", module)
        table_name = _require_arg("table", table)
        payload = {
            "tool": tool,
            "run_dir": str(run_root),
            "table": read_analysis_table(run_root, module_name, table_name, limit=limit),
        }
    elif tool == "analysis.read_plot_spec":
        module_name = _require_arg("module", module)
        plot_name = _require_arg("plot", plot)
        payload = {
            "tool": tool,
            "run_dir": str(run_root),
            "plot": read_analysis_plot_spec(run_root, module_name, plot_name),
        }
    elif tool == "cases.list_failures":
        module_name = module or "prediction_audit"
        payload = {"tool": tool, "run_dir": str(run_root), "cases": list_failure_cases(run_root, module_name=module_name)}
    elif tool == "cases.read_failures":
        module_name = module or "prediction_audit"
        payload = {
            "tool": tool,
            "run_dir": str(run_root),
            "cases": read_failure_cases(run_root, module_name=module_name, name=name, limit=limit),
        }
    elif tool == "cases.describe_patient":
        module_name = module or "prediction_audit"
        patient = _require_arg("patient-id", patient_id)
        payload = {
            "tool": tool,
            "run_dir": str(run_root),
            "patient": describe_patient_case(run_root, patient, module_name=module_name, limit=limit),
        }
    elif tool == "cohorts.compare":
        payload = {
            "tool": tool,
            "run_dir": str(run_root),
            "comparison": compare_cohorts(
                run_root,
                split=split,
                left_role=left_role,
                right_role=right_role,
                top_k=top_k,
            ),
        }
    else:  # pragma: no cover
        raise SystemExit(f"Unhandled inspect tool {tool!r}")

    sys.stdout.write(json.dumps(as_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _resolve_run_root(*, config: str | None, run_dir: str | None, tool: str) -> Path:
    if tool == "runs.list":
        return Path(run_dir or ".")
    if run_dir is not None:
        return Path(run_dir)
    if config is None:
        raise SystemExit(f"{tool} requires --run-dir or --config")
    from oneehr.cli._common import resolve_run_root
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(config)
    return resolve_run_root(cfg, None)


def _require_arg(name: str, value: str | None) -> str:
    if value in {None, ""}:
        raise SystemExit(f"--{name} is required for this inspect tool")
    return str(value)
