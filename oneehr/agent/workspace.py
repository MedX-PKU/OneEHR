from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from oneehr.analysis.read import describe_patient_case, read_analysis_index
from oneehr.artifacts.run_io import RunIO
from oneehr.config.schema import ExperimentConfig
from oneehr.data.io import load_dynamic_table_optional, load_static_table
from oneehr.data.patient_index import make_patient_index, make_patient_index_from_static
from oneehr.data.splits import Split, load_splits, make_splits, save_splits
from oneehr.llm.templates import safe_case_slug, select_events
from oneehr.utils.io import as_jsonable, ensure_dir, write_json


@dataclass(frozen=True)
class MaterializedWorkspace:
    index_path: Path
    case_count: int


def materialize_case_workspaces(cfg: ExperimentConfig, *, run_root: Path, force: bool = False) -> MaterializedWorkspace:
    workspace_root = run_root / "workspace"
    index_path = workspace_root / "index.json"
    if force and workspace_root.exists():
        for path in sorted(workspace_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    ensure_dir(workspace_root)

    cases = _build_cases(cfg, run_root=run_root)
    dynamic_by_patient = _load_dynamic_by_patient(cfg)
    static_by_patient = _load_static_by_patient(cfg) if cfg.workspace.include_static else {}
    case_records: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case["case_id"])
        case_dir = ensure_dir(workspace_root / "cases" / _case_dir_name(case_id))
        case_payload = dict(case)
        patient_id = str(case["patient_id"])
        anchor_time = _optional_timestamp(case.get("bin_time"))
        events = select_events(
            dynamic=dynamic_by_patient.get(patient_id),
            anchor_time=anchor_time,
            history_window=cfg.workspace.history_window,
            max_events=cfg.workspace.max_events,
            time_order=cfg.workspace.time_order,
        )
        static_row = static_by_patient.get(patient_id)
        predictions = _collect_case_predictions(run_root=run_root, case=case)
        analysis_refs = (
            _collect_analysis_refs(run_root=run_root, patient_id=patient_id)
            if cfg.workspace.include_analysis_refs
            else {"modules": [], "patient_case_matches": []}
        )

        events_path = case_dir / "events.csv"
        static_path = case_dir / "static.json"
        preds_path = case_dir / "predictions.csv"
        refs_path = case_dir / "analysis_refs.json"
        workspace_path = case_dir / "workspace.json"

        if events.empty:
            pd.DataFrame(columns=["patient_id", "event_time", "code", "value"]).to_csv(events_path, index=False)
        else:
            out_events = events.copy()
            out_events["patient_id"] = patient_id
            out_events[["patient_id", "event_time", "code", "value"]].to_csv(events_path, index=False)
        write_json(static_path, {"patient_id": patient_id, "features": _series_to_dict(static_row)})
        if predictions.empty:
            pd.DataFrame(
                columns=[
                    "source",
                    "model_name",
                    "split",
                    "patient_id",
                    "prediction",
                    "probability",
                    "value",
                    "confidence",
                    "explanation",
                    "parsed_ok",
                    "error_code",
                    "ground_truth",
                ]
            ).to_csv(preds_path, index=False)
        else:
            predictions.to_csv(preds_path, index=False)
        write_json(refs_path, analysis_refs)

        case_payload["artifacts"] = {
            "workspace_json": str(workspace_path.relative_to(run_root)),
            "events_csv": str(events_path.relative_to(run_root)),
            "static_json": str(static_path.relative_to(run_root)),
            "predictions_csv": str(preds_path.relative_to(run_root)),
            "analysis_refs_json": str(refs_path.relative_to(run_root)),
        }
        case_payload["evidence"] = {
            "anchor_time": None if anchor_time is None else anchor_time.isoformat(),
            "event_count": int(len(events)),
            "static_feature_count": int(len(_series_to_dict(static_row))),
            "prediction_count": int(len(predictions)),
            "analysis_module_count": int(len(analysis_refs.get("modules", []))),
            "patient_case_match_count": int(len(analysis_refs.get("patient_case_matches", []))),
        }
        write_json(workspace_path, as_jsonable(case_payload))

        case_records.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "split": str(case["split"]),
                "prediction_mode": str(case["prediction_mode"]),
                "bin_time": None if anchor_time is None else anchor_time.isoformat(),
                "ground_truth": case.get("ground_truth"),
                "workspace_path": str(workspace_path.relative_to(run_root)),
                "event_count": int(len(events)),
                "prediction_count": int(len(predictions)),
            }
        )

    write_json(
        index_path,
        {
            "schema_version": 1,
            "run_dir": str(run_root),
            "case_count": int(len(case_records)),
            "records": case_records,
        },
    )
    return MaterializedWorkspace(index_path=index_path, case_count=len(case_records))


def read_workspace_index(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root) / "workspace" / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing workspace index: {path}. Run `oneehr workspace` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_workspace_cases(run_root: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    records = read_workspace_index(run_root).get("records", [])
    if not isinstance(records, list):
        return []
    rows = [row for row in records if isinstance(row, dict)]
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def read_workspace_case(run_root: str | Path, case_id: str, *, limit: int | None = None) -> dict[str, Any]:
    run_root = Path(run_root)
    records = list_workspace_cases(run_root)
    match = next((row for row in records if str(row.get("case_id")) == str(case_id)), None)
    if match is None:
        raise FileNotFoundError(f"Unknown workspace case_id {case_id!r} under {run_root}")

    workspace_path = run_root / str(match["workspace_path"])
    payload = json.loads(workspace_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {})
    events = pd.read_csv(run_root / str(artifacts["events_csv"])) if artifacts.get("events_csv") else pd.DataFrame()
    predictions = (
        pd.read_csv(run_root / str(artifacts["predictions_csv"]))
        if artifacts.get("predictions_csv")
        else pd.DataFrame()
    )
    if limit is not None:
        events = events.head(int(limit)).reset_index(drop=True)
        predictions = predictions.head(int(limit)).reset_index(drop=True)
    static_payload = (
        json.loads((run_root / str(artifacts["static_json"])).read_text(encoding="utf-8"))
        if artifacts.get("static_json")
        else {"patient_id": payload.get("patient_id"), "features": {}}
    )
    refs_payload = (
        json.loads((run_root / str(artifacts["analysis_refs_json"])).read_text(encoding="utf-8"))
        if artifacts.get("analysis_refs_json")
        else {"modules": [], "patient_case_matches": []}
    )
    payload["events"] = events.to_dict(orient="records")
    payload["predictions"] = predictions.to_dict(orient="records")
    payload["static"] = static_payload
    payload["analysis_refs"] = refs_payload
    return payload


def _build_cases(cfg: ExperimentConfig, *, run_root: Path) -> list[dict[str, Any]]:
    run = RunIO(run_root=run_root)
    manifest = run.require_manifest()
    labels_df = run.load_labels(manifest)
    splits = _ensure_case_splits(cfg, run_root=run_root)

    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    static = load_static_table(train_dataset.static)

    if cfg.task.prediction_mode == "patient":
        cases = _build_patient_cases(splits=splits, labels_df=labels_df, dynamic=dynamic, static=static, task_kind=cfg.task.kind)
    else:
        cases = _build_time_cases(splits=splits, labels_df=labels_df, dynamic=dynamic, static=static, task_kind=cfg.task.kind)

    if cfg.workspace.case_limit is not None:
        cases = cases[: int(cfg.workspace.case_limit)]
    return cases


def _ensure_case_splits(cfg: ExperimentConfig, *, run_root: Path) -> list[Split]:
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
        raise SystemExit("dataset.dynamic or dataset.static is required to materialize workspace splits.")

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


def _build_patient_cases(
    *,
    splits: list[Split],
    labels_df: pd.DataFrame | None,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    task_kind: str,
) -> list[dict[str, Any]]:
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
                "first_event_time": pd.to_datetime(grp["event_time"]).min(),
                "last_event_time": pd.to_datetime(grp["event_time"]).max(),
            }
            for pid, grp in grouped
        }

    static_ids = set()
    if static is not None and not static.empty:
        static_ids = set(static["patient_id"].astype(str).tolist())

    rows: list[dict[str, Any]] = []
    for sp in splits:
        for patient_id in sp.test_patients.astype(str).tolist():
            stats = dyn_stats.get(patient_id, {})
            rows.append(
                {
                    "case_id": f"{sp.name}:{patient_id}",
                    "patient_id": patient_id,
                    "split": sp.name,
                    "split_role": "test",
                    "prediction_mode": "patient",
                    "task_kind": task_kind,
                    "ground_truth": label_map.get(patient_id),
                    "first_event_time": stats.get("first_event_time"),
                    "last_event_time": stats.get("last_event_time"),
                    "has_static": patient_id in static_ids,
                }
            )
    rows = sorted(rows, key=lambda row: (str(row["split"]), str(row["patient_id"])))
    return rows


def _build_time_cases(
    *,
    splits: list[Split],
    labels_df: pd.DataFrame | None,
    dynamic: pd.DataFrame | None,
    static: pd.DataFrame | None,
    task_kind: str,
) -> list[dict[str, Any]]:
    if dynamic is None or dynamic.empty:
        raise SystemExit("Workspace time-window cases require dataset.dynamic.")
    if labels_df is None or labels_df.empty:
        raise SystemExit("Workspace time-window cases require labels.parquet.")

    static_ids = set()
    if static is not None and not static.empty:
        static_ids = set(static["patient_id"].astype(str).tolist())

    labels = labels_df.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels["bin_time"] = pd.to_datetime(labels["bin_time"], errors="raise")
    if "mask" in labels.columns:
        labels = labels[labels["mask"].astype(int) != 0].copy()

    rows: list[dict[str, Any]] = []
    for sp in splits:
        test_patients = set(sp.test_patients.astype(str).tolist())
        block = labels[labels["patient_id"].isin(test_patients)].copy()
        block = block.sort_values(["patient_id", "bin_time"], kind="stable")
        for _, row in block.iterrows():
            patient_id = str(row["patient_id"])
            bin_time = pd.to_datetime(row["bin_time"], errors="raise")
            rows.append(
                {
                    "case_id": f"{sp.name}:{patient_id}:{bin_time.isoformat()}",
                    "patient_id": patient_id,
                    "split": sp.name,
                    "split_role": "test",
                    "prediction_mode": "time",
                    "task_kind": task_kind,
                    "bin_time": bin_time,
                    "ground_truth": float(row["label"]),
                    "has_static": patient_id in static_ids,
                }
            )
    return rows


def _load_dynamic_by_patient(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    dynamic = load_dynamic_table_optional(train_dataset.dynamic)
    if dynamic is None or dynamic.empty:
        return {}
    tmp = dynamic.copy()
    tmp["patient_id"] = tmp["patient_id"].astype(str)
    tmp["event_time"] = pd.to_datetime(tmp["event_time"], errors="raise")
    return {
        str(pid): grp.sort_values("event_time", kind="stable").reset_index(drop=True)
        for pid, grp in tmp.groupby("patient_id", sort=False)
    }


def _load_static_by_patient(cfg: ExperimentConfig) -> dict[str, pd.Series]:
    train_dataset = cfg.datasets.train if cfg.datasets is not None else cfg.dataset
    static = load_static_table(train_dataset.static)
    if static is None or static.empty:
        return {}
    tmp = static.copy()
    tmp["patient_id"] = tmp["patient_id"].astype(str)
    return {
        str(row["patient_id"]): row
        for _, row in tmp.drop_duplicates(subset=["patient_id"], keep="last").iterrows()
    }


def _collect_case_predictions(*, run_root: Path, case: dict[str, Any]) -> pd.DataFrame:
    split = str(case["split"])
    patient_id = str(case["patient_id"])
    bin_time = _optional_timestamp(case.get("bin_time"))
    frames: list[pd.DataFrame] = []

    train_preds_root = run_root / "preds"
    if train_preds_root.exists():
        for model_dir in sorted(train_preds_root.iterdir(), key=lambda p: p.name):
            if not model_dir.is_dir():
                continue
            path = model_dir / f"{split}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path).copy()
            if "patient_id" not in df.columns:
                continue
            df["patient_id"] = df["patient_id"].astype(str)
            block = df[df["patient_id"] == patient_id].copy()
            if block.empty:
                continue
            if bin_time is not None and "bin_time" in block.columns:
                block["bin_time"] = pd.to_datetime(block["bin_time"], errors="raise")
                block = block[block["bin_time"] == bin_time].copy()
            elif bin_time is not None:
                continue
            if block.empty:
                continue
            block["source"] = "train"
            block["model_name"] = model_dir.name
            block["split"] = split
            block["prediction"] = block.get("y_pred")
            block["probability"] = block.get("y_pred")
            block["value"] = block.get("y_pred")
            block["confidence"] = None
            block["explanation"] = None
            block["parsed_ok"] = True
            block["error_code"] = None
            block["ground_truth"] = block.get("y_true")
            keep = [
                "source",
                "model_name",
                "split",
                "patient_id",
                "prediction",
                "probability",
                "value",
                "confidence",
                "explanation",
                "parsed_ok",
                "error_code",
                "ground_truth",
            ]
            if "bin_time" in block.columns:
                keep.insert(4, "bin_time")
            frames.append(block[keep])

    llm_preds_root = run_root / "llm" / "preds"
    if llm_preds_root.exists():
        for model_dir in sorted(llm_preds_root.iterdir(), key=lambda p: p.name):
            if not model_dir.is_dir():
                continue
            path = model_dir / f"{split}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path).copy()
            if "patient_id" not in df.columns:
                continue
            df["patient_id"] = df["patient_id"].astype(str)
            block = df[df["patient_id"] == patient_id].copy()
            if block.empty:
                continue
            if bin_time is not None and "bin_time" in block.columns:
                block["bin_time"] = pd.to_datetime(block["bin_time"], errors="raise")
                block = block[block["bin_time"] == bin_time].copy()
            elif bin_time is not None:
                continue
            if block.empty:
                continue
            block["source"] = "llm"
            block["model_name"] = model_dir.name
            keep = [
                "source",
                "model_name",
                "split",
                "patient_id",
                "prediction",
                "probability",
                "value",
                "confidence",
                "explanation",
                "parsed_ok",
                "error_code",
                "ground_truth",
            ]
            if "bin_time" in block.columns:
                keep.insert(4, "bin_time")
            frames.append(block[keep])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "bin_time" in out.columns:
        out["bin_time"] = pd.to_datetime(out["bin_time"], errors="coerce")
    return out.sort_values([col for col in ("source", "model_name", "patient_id", "bin_time") if col in out.columns], kind="stable").reset_index(drop=True)


def _collect_analysis_refs(*, run_root: Path, patient_id: str) -> dict[str, Any]:
    index: dict[str, Any] | None = None
    try:
        index = read_analysis_index(run_root)
    except FileNotFoundError:
        index = {"modules": []}

    matches: list[dict[str, Any]] = []
    for module_name in ("prediction_audit", "llm_audit"):
        try:
            desc = describe_patient_case(run_root, patient_id, module_name=module_name)
        except FileNotFoundError:
            continue
        for item in list(desc.get("matches", [])):
            if isinstance(item, dict):
                matches.append(
                    {
                        "module": module_name,
                        "name": str(item.get("split", "case")),
                        "row_count": 1,
                        "record": item,
                    }
                )
    return {
        "modules": list(index.get("modules", [])) if isinstance(index, dict) else [],
        "patient_case_matches": matches,
    }


def _series_to_dict(row: pd.Series | None) -> dict[str, Any]:
    if row is None or row.empty:
        return {}
    out: dict[str, Any] = {}
    for key, value in row.items():
        if str(key) == "patient_id" or pd.isna(value):
            continue
        out[str(key)] = value
    return as_jsonable(out)


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value in {None, "", "NaT"}:
        return None
    return pd.to_datetime(value, errors="raise")


def _case_dir_name(case_id: str) -> str:
    base = safe_case_slug(case_id)[:80]
    suffix = hashlib.sha1(str(case_id).encode("utf-8")).hexdigest()[:10]
    return f"{base}_{suffix}"
