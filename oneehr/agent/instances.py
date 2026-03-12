from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oneehr.agent.templates import get_prompt_template
from oneehr.artifacts.run_io import RunIO
from oneehr.config.schema import ExperimentConfig
from oneehr.data.io import load_dynamic_table_optional, load_static_table
from oneehr.data.patient_index import make_patient_index, make_patient_index_from_static
from oneehr.data.splits import Split, load_splits, make_splits, save_splits
from oneehr.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class MaterializedAgentInstances:
    path: Path
    frame: pd.DataFrame


def validate_agent_predict_setup(cfg: ExperimentConfig) -> None:
    predict_cfg = cfg.agent.predict
    if not predict_cfg.enabled:
        raise SystemExit("Agent prediction workflow is disabled. Set agent.predict.enabled = true in the config.")
    if cfg.task.kind not in {"binary", "regression"}:
        raise SystemExit("Agent prediction currently supports task.kind = 'binary' or 'regression' only.")
    if predict_cfg.sample_unit != cfg.task.prediction_mode:
        raise SystemExit(
            "agent.predict.sample_unit must match task.prediction_mode to keep evaluation semantics consistent."
        )
    try:
        template = get_prompt_template(predict_cfg.prompt_template)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    if template.family != "prediction":
        raise SystemExit(
            f"agent.predict.prompt_template must resolve to a prediction template, got {template.family!r}."
        )
    if cfg.task.kind not in set(template.supported_task_kinds):
        raise SystemExit(
            f"agent.predict.prompt_template={predict_cfg.prompt_template!r} does not support "
            f"task.kind={cfg.task.kind!r}."
        )
    if predict_cfg.sample_unit not in set(template.supported_sample_units):
        raise SystemExit(
            f"agent.predict.prompt_template={predict_cfg.prompt_template!r} does not support "
            f"sample_unit={predict_cfg.sample_unit!r}."
        )
    if predict_cfg.json_schema_version not in set(template.supported_schema_versions):
        raise SystemExit(
            f"agent.predict.prompt_template={predict_cfg.prompt_template!r} does not support "
            f"json_schema_version={predict_cfg.json_schema_version!r}."
        )
    if predict_cfg.prompt.include_labels_context and not template.allow_labels_context:
        raise SystemExit(
            "agent.predict.prompt.include_labels_context is not allowed for "
            f"prompt template {predict_cfg.prompt_template!r}."
        )
    if not predict_cfg.backends:
        raise SystemExit("At least one [[agent.predict.backends]] entry is required for agent prediction.")


def agent_instance_path(run_root: Path, sample_unit: str) -> Path:
    base = ensure_dir(run_root / "agent" / "predict" / "instances")
    if sample_unit == "patient":
        return base / "patient_instances.parquet"
    if sample_unit == "time":
        return base / "time_instances.parquet"
    raise ValueError(f"Unsupported agent sample_unit: {sample_unit!r}")


def ensure_agent_predict_splits(cfg: ExperimentConfig, *, run_root: Path) -> list[Split]:
    split_dir = run_root / "splits"
    splits = load_splits(split_dir)
    if splits:
        return splits

    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static = load_static_table(train_dataset.static)
    if dynamic is not None:
        patient_index = make_patient_index(dynamic, "event_time", "patient_id")
    elif static is not None:
        patient_index = make_patient_index_from_static(static, patient_id_col="patient_id")
    else:
        raise SystemExit("dataset.dynamic or dataset.static is required to materialize agent prediction splits.")

    splits = make_splits(patient_index, cfg.split)
    if cfg.trainer.repeat > 1:
        expanded: list[Split] = []
        for sp in splits:
            for repeat_idx in range(cfg.trainer.repeat):
                expanded.append(
                    Split(
                        name=f"{sp.name}__r{repeat_idx}",
                        train_patients=sp.train_patients,
                        val_patients=sp.val_patients,
                        test_patients=sp.test_patients,
                    )
                )
        splits = expanded
    save_splits(splits, split_dir)
    return splits


def materialize_agent_instances(cfg: ExperimentConfig, *, run_root: Path) -> MaterializedAgentInstances:
    validate_agent_predict_setup(cfg)

    run = RunIO(run_root=run_root)
    manifest = run.require_manifest()
    labels_df = run.load_labels(manifest)
    splits = ensure_agent_predict_splits(cfg, run_root=run_root)

    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static = load_static_table(train_dataset.static)

    if cfg.agent.predict.sample_unit == "patient":
        frame = _build_patient_instances(
            splits=splits,
            labels_df=labels_df,
            dynamic=dynamic,
            static=static,
            task_kind=cfg.task.kind,
        )
    else:
        frame = _build_time_instances(
            splits=splits,
            labels_df=labels_df,
            dynamic=dynamic,
            task_kind=cfg.task.kind,
        )

    sort_cols = ["split", "patient_id"]
    if "bin_time" in frame.columns:
        sort_cols.append("bin_time")
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    if cfg.agent.predict.max_samples is not None:
        frame = frame.head(int(cfg.agent.predict.max_samples)).reset_index(drop=True)

    path = agent_instance_path(run_root, cfg.agent.predict.sample_unit)
    ensure_dir(path.parent)
    frame.to_parquet(path, index=False)
    write_json(
        path.parent / "summary.json",
        {
            "sample_unit": cfg.agent.predict.sample_unit,
            "rows": int(len(frame)),
            "splits": sorted(frame["split"].astype(str).unique().tolist()) if not frame.empty else [],
        },
    )
    return MaterializedAgentInstances(path=path, frame=frame)


def _build_patient_instances(
    *,
    splits: list[Split],
    labels_df: pd.DataFrame | None,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    task_kind: str,
) -> pd.DataFrame:
    label_map: dict[str, float] = {}
    if labels_df is not None and not labels_df.empty:
        labels = labels_df[["patient_id", "label"]].copy()
        labels["patient_id"] = labels["patient_id"].astype(str)
        labels = labels.drop_duplicates(subset=["patient_id"], keep="last")
        label_map = {str(pid): float(lbl) for pid, lbl in zip(labels["patient_id"], labels["label"])}

    dyn_stats: dict[str, dict[str, object]] = {}
    if dynamic is not None and not dynamic.empty:
        tmp = dynamic.copy()
        tmp["patient_id"] = tmp["patient_id"].astype(str)
        grouped = tmp.groupby("patient_id", sort=False)
        dyn_stats = {
            str(pid): {
                "event_count": int(len(group)),
                "first_event_time": pd.to_datetime(group["event_time"]).min(),
                "last_event_time": pd.to_datetime(group["event_time"]).max(),
            }
            for pid, group in grouped
        }

    static_ids = set()
    if static is not None and not static.empty:
        static_ids = set(static["patient_id"].astype(str).tolist())

    rows: list[dict[str, object]] = []
    for sp in splits:
        for patient_id in sp.test_patients.astype(str).tolist():
            stats = dyn_stats.get(patient_id, {})
            rows.append(
                {
                    "instance_id": f"{sp.name}:{patient_id}",
                    "split": sp.name,
                    "split_role": "test",
                    "patient_id": patient_id,
                    "task_kind": task_kind,
                    "ground_truth": label_map.get(patient_id),
                    "event_count": int(stats.get("event_count", 0)),
                    "first_event_time": stats.get("first_event_time"),
                    "last_event_time": stats.get("last_event_time"),
                    "has_static": patient_id in static_ids,
                }
            )
    return pd.DataFrame(rows)


def _build_time_instances(
    *,
    splits: list[Split],
    labels_df: pd.DataFrame | None,
    dynamic: pd.DataFrame | None,
    task_kind: str,
) -> pd.DataFrame:
    if dynamic is None or dynamic.empty:
        raise SystemExit("Agent time-window prediction requires dataset.dynamic.")
    if labels_df is None or labels_df.empty:
        raise SystemExit("Agent time-window prediction requires labels.fn / labels.parquet.")

    labels = labels_df.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels["bin_time"] = pd.to_datetime(labels["bin_time"], errors="raise")
    if "mask" in labels.columns:
        labels = labels[labels["mask"].astype(int) != 0].copy()

    rows: list[dict[str, object]] = []
    for sp in splits:
        test_patients = set(sp.test_patients.astype(str).tolist())
        split_labels = labels[labels["patient_id"].isin(test_patients)].copy()
        split_labels = split_labels.sort_values(["patient_id", "bin_time"], kind="stable")
        for _, row in split_labels.iterrows():
            patient_id = str(row["patient_id"])
            bin_time = pd.to_datetime(row["bin_time"], errors="raise")
            rows.append(
                {
                    "instance_id": f"{sp.name}:{patient_id}:{bin_time.isoformat()}",
                    "split": sp.name,
                    "split_role": "test",
                    "patient_id": patient_id,
                    "bin_time": bin_time,
                    "task_kind": task_kind,
                    "ground_truth": float(row["label"]),
                }
            )
    return pd.DataFrame(rows)
