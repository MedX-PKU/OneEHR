"""Tests for the train CLI contract."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def test_run_train_requires_preprocess(tmp_path: Path) -> None:
    """Train should fail if preprocess hasn't been run."""
    cfg_path = tmp_path / "exp.toml"
    cfg_path.write_text("""
[dataset]
dynamic = "dummy.csv"

[task]
kind = "binary"

[split]
kind = "random"

[[models]]
name = "xgboost"

[output]
root = "{root}"
run_name = "test"
""".format(root=tmp_path / "runs"), encoding="utf-8")

    from oneehr.cli.train import run_train

    with pytest.raises(SystemExit, match="Preprocessed artifacts not found"):
        run_train(str(cfg_path), force=False)


def test_run_train_requires_force_for_existing(tmp_path: Path) -> None:
    """Train should fail if train dir exists and --force not set."""
    out = tmp_path / "runs" / "test"
    (out / "preprocess").mkdir(parents=True)
    (out / "train").mkdir(parents=True)
    # Write minimal manifest
    (out / "manifest.json").write_text(json.dumps({
        "config": {}, "feature_columns": [], "static_feature_columns": [],
        "paths": {"binned": "preprocess/binned.parquet", "labels": "preprocess/labels.parquet", "split": "preprocess/split.json"},
    }), encoding="utf-8")

    cfg_path = tmp_path / "exp.toml"
    cfg_path.write_text(f"""
[dataset]
dynamic = "dummy.csv"

[task]
kind = "binary"

[split]
kind = "random"

[[models]]
name = "xgboost"

[output]
root = "{tmp_path / 'runs'}"
run_name = "test"
""", encoding="utf-8")

    from oneehr.cli.train import run_train

    with pytest.raises(SystemExit, match="Use --force"):
        run_train(str(cfg_path), force=False)
