"""oneehr llm-preprocess subcommand."""
from __future__ import annotations

import shutil

from oneehr.cli._common import resolve_run_root
from oneehr.llm.instances import materialize_llm_instances, validate_llm_setup


def run_llm_preprocess(cfg_path: str, *, run_dir: str | None, force: bool) -> None:
    from oneehr.config.load import load_experiment_config

    cfg = load_experiment_config(cfg_path)
    validate_llm_setup(cfg)
    run_root = resolve_run_root(cfg, run_dir)
    if not run_root.exists():
        raise SystemExit(
            f"Run directory not found: {run_root}. "
            "Run `oneehr preprocess` first."
        )

    llm_root = run_root / "llm" / "instances"
    if llm_root.exists() and force:
        shutil.rmtree(llm_root)
    materialize_llm_instances(cfg, run_root=run_root)

