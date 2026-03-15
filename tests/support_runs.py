from __future__ import annotations

import os
import subprocess
from pathlib import Path

from test_analysis import _build_trained_run as build_trained_run
from test_eval_workflow import _append_eval_config, _mock_eval_server
from test_inspect import _build_analyzed_run as build_analyzed_run


def build_eval_run(*, tmp_path: Path, run_name: str, seed: int) -> tuple[Path, Path]:
    with _mock_eval_server() as (_, base_url):
        run_root, cfg_path = build_trained_run(tmp_path=tmp_path, run_name=run_name, seed=seed)
        _append_eval_config(cfg_path, base_url=base_url)
        env = os.environ.copy()
        env["TEST_OPENAI_API_KEY"] = "dummy"
        subprocess.check_call(["oneehr", "eval", "build", "--config", str(cfg_path), "--force"], env=env)
        subprocess.check_call(["oneehr", "eval", "run", "--config", str(cfg_path), "--force"], env=env)
        subprocess.check_call(["oneehr", "eval", "report", "--config", str(cfg_path), "--force"], env=env)
        return run_root, cfg_path


__all__ = [
    "build_analyzed_run",
    "build_eval_run",
    "build_trained_run",
]
