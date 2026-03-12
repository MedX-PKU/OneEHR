"""oneehr workspace subcommand."""
from __future__ import annotations

from oneehr.cli._common import resolve_run_root
from oneehr.agent.workspace import materialize_case_workspaces


def run_workspace(cfg_path: str, *, run_dir: str | None, force: bool) -> None:
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(cfg_path)
    run_root = resolve_run_root(cfg, run_dir)
    if not run_root.exists():
        raise SystemExit(
            f"Run directory not found: {run_root}. "
            "Run `oneehr preprocess` first."
        )
    materialize_case_workspaces(cfg, run_root=run_root, force=force)
