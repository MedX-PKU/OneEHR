"""Shared helpers for CLI subcommands."""

from __future__ import annotations

from pathlib import Path

from oneehr.config.schema import ExperimentConfig


def resolve_run_root(cfg: ExperimentConfig, run_dir: str | None) -> Path:
    """Resolve the run root directory from an explicit *run_dir* or config."""
    if run_dir is not None:
        return Path(run_dir)
    return cfg.output.root / cfg.output.run_name


def require_manifest(run_root: Path):
    """Load the run manifest or exit with a helpful message.

    Returns an ``RunManifest`` instance.
    """
    from oneehr.artifacts.manifest import read_run_manifest

    manifest = read_run_manifest(run_root)
    if manifest is None:
        raise SystemExit(
            f"Missing run_manifest.json under {run_root}. "
            "Run `oneehr preprocess` first."
        )
    return manifest
