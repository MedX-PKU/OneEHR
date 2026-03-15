from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_trace_rows(
    run_root: str | Path,
    *,
    system_name: str,
    limit: int = 25,
    offset: int = 0,
    stage: str | None = None,
    role: str | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root)
    trace_path = run_root / "eval" / "traces" / str(system_name) / "trace.parquet"
    if not trace_path.exists():
        return {
            "system_name": str(system_name),
            "total_rows": 0,
            "row_count": 0,
            "offset": int(offset),
            "limit": int(limit),
            "records": [],
        }
    df = pd.read_parquet(trace_path)
    if stage is not None and "stage" in df.columns:
        df = df[df["stage"].astype(str) == str(stage)].copy()
    if role is not None and "role" in df.columns:
        df = df[df["role"].astype(str) == str(role)].copy()
    if round_index is not None and "round" in df.columns:
        df = df[pd.to_numeric(df["round"], errors="coerce") == int(round_index)].copy()
    total_rows = int(len(df))
    page = df.iloc[int(offset) : int(offset) + int(limit)].copy()
    page = page.astype(object).where(pd.notna(page), None)
    return {
        "system_name": str(system_name),
        "total_rows": total_rows,
        "row_count": int(len(page)),
        "offset": int(offset),
        "limit": int(limit),
        "records": page.to_dict(orient="records"),
    }


def read_instance_payload(run_root: str | Path, *, instance_id: str) -> dict[str, Any]:
    run_root = Path(run_root)
    index_path = run_root / "eval" / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing eval index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record = next(
        (
            item
            for item in index.get("records", [])
            if isinstance(item, dict) and str(item.get("instance_id")) == str(instance_id)
        ),
        None,
    )
    if record is None:
        raise FileNotFoundError(f"Unknown instance_id {instance_id!r}")

    evidence = {
        key: value
        for key, value in record.items()
        if key not in {"evidence_path"}
    }
    if record.get("evidence_path"):
        evidence_path = run_root / str(record["evidence_path"])
        evidence_meta = json.loads(evidence_path.read_text(encoding="utf-8"))
        artifacts = evidence_meta.get("artifacts") or {}
        events_path = run_root / str(artifacts.get("events_csv"))
        static_path = run_root / str(artifacts.get("static_json"))
        refs_path = run_root / str(artifacts.get("analysis_refs_json"))
        if events_path.exists():
            events_df = pd.read_csv(events_path)
            events_df = events_df.astype(object).where(pd.notna(events_df), None)
            evidence["events"] = events_df.to_dict(orient="records")
        else:
            evidence["events"] = []
        evidence["static"] = json.loads(static_path.read_text(encoding="utf-8")) if static_path.exists() else {"features": {}}
        evidence["analysis_refs"] = json.loads(refs_path.read_text(encoding="utf-8")) if refs_path.exists() else {"modules": [], "patient_case_matches": []}
    else:
        evidence["events"] = []
        evidence["static"] = {"features": {}}
        evidence["analysis_refs"] = {"modules": [], "patient_case_matches": []}

    outputs: list[dict[str, Any]] = []
    pred_root = run_root / "eval" / "predictions"
    if pred_root.exists():
        for pred_path in sorted(pred_root.glob("*/predictions.parquet")):
            df = pd.read_parquet(pred_path)
            block = df[df["instance_id"].astype(str) == str(instance_id)].copy()
            if block.empty:
                continue
            block = block.astype(object).where(pd.notna(block), None)
            outputs.extend(block.to_dict(orient="records"))

    return {
        "instance_id": str(instance_id),
        "evidence": evidence,
        "outputs": outputs,
    }
