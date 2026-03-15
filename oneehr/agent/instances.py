from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oneehr.agent.validation import validate_agent_predict_setup
from oneehr.artifacts.run_io import RunIO
from oneehr.config.schema import ExperimentConfig
from oneehr.data.io import load_dynamic_table_optional, load_static_table
from oneehr.data.splits import require_saved_splits
from oneehr.data.test_samples import build_test_sample_frame
from oneehr.utils.io import ensure_dir


@dataclass(frozen=True)
class MaterializedAgentInstances:
    path: Path
    frame: pd.DataFrame


def agent_instance_path(run_root: Path, sample_unit: str) -> Path:
    base = ensure_dir(run_root / "agent" / "predict" / "instances")
    if sample_unit == "patient":
        return base / "patient_instances.parquet"
    if sample_unit == "time":
        return base / "time_instances.parquet"
    raise ValueError(f"Unsupported agent sample_unit: {sample_unit!r}")


def materialize_agent_instances(cfg: ExperimentConfig, *, run_root: Path) -> MaterializedAgentInstances:
    validate_agent_predict_setup(cfg)

    run = RunIO(run_root=run_root)
    manifest = run.require_manifest()
    labels_df = run.load_labels(manifest)
    splits = require_saved_splits(
        run_root / "splits",
        context="running `oneehr agent predict`",
    )

    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static = load_static_table(train_dataset.static)
    frame = build_test_sample_frame(
        splits=splits,
        labels_df=labels_df,
        dynamic=dynamic,
        static=static,
        task_kind=cfg.task.kind,
        prediction_mode=cfg.agent.predict.sample_unit,
    )
    if frame.empty:
        frame = pd.DataFrame(columns=["sample_id", "split", "split_role", "patient_id"])

    sort_cols = ["split", "patient_id"]
    if "bin_time" in frame.columns:
        sort_cols.append("bin_time")
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    if cfg.agent.predict.max_samples is not None:
        frame = frame.head(int(cfg.agent.predict.max_samples)).reset_index(drop=True)

    path = agent_instance_path(run_root, cfg.agent.predict.sample_unit)
    ensure_dir(path.parent)
    out = frame.rename(columns={"sample_id": "instance_id"})
    out.to_parquet(path, index=False)
    return MaterializedAgentInstances(path=path, frame=out)
