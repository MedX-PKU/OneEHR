"""Run manifest: config snapshot + feature columns + artifact paths."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from oneehr.config.schema import ExperimentConfig
from oneehr.utils import as_jsonable, ensure_dir, write_json


def write_manifest(
    *,
    out_dir: Path,
    cfg: ExperimentConfig,
    feature_columns: list[str],
    static_feature_columns: list[str] | None = None,
) -> None:
    out_dir = ensure_dir(out_dir)
    manifest = {
        "config": as_jsonable(asdict(cfg)),
        "feature_columns": list(feature_columns),
        "static_feature_columns": list(static_feature_columns or []),
        "paths": {
            "binned": "preprocess/binned.parquet",
            "labels": "preprocess/labels.parquet",
            "split": "preprocess/split.json",
            "static": "preprocess/static.parquet" if static_feature_columns else None,
            "feature_schema": "preprocess/feature_schema.json",
            "obs_mask": "preprocess/obs_mask.parquet",
        },
    }
    write_json(out_dir / "manifest.json", manifest)


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(
            f"Missing manifest.json at {run_dir}. Run `oneehr preprocess` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
