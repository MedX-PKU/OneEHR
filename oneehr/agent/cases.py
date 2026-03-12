from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AgentCaseContext:
    case_id: str
    patient_id: str
    dynamic: pd.DataFrame
    static_row: pd.Series | None
    instance: dict[str, object]


def case_static_features(case: dict[str, Any]) -> dict[str, object]:
    static_payload = case.get("static", {})
    if not isinstance(static_payload, dict):
        return {}
    maybe = static_payload.get("features", {})
    if not isinstance(maybe, dict):
        return {}
    return {str(key): value for key, value in maybe.items()}


def build_case_static_row(case: dict[str, Any]) -> pd.Series | None:
    features = case_static_features(case)
    if not features:
        return None
    return pd.Series({"patient_id": str(case.get("patient_id")), **features})


def build_case_context(
    case: dict[str, Any],
    *,
    default_prediction_mode: str,
    include_ground_truth: bool,
) -> AgentCaseContext:
    case_id = str(case.get("case_id"))
    patient_id = str(case.get("patient_id"))
    return AgentCaseContext(
        case_id=case_id,
        patient_id=patient_id,
        dynamic=pd.DataFrame(case.get("events", [])),
        static_row=build_case_static_row(case),
        instance={
            "case_id": case_id,
            "instance_id": case_id,
            "patient_id": patient_id,
            "split": case.get("split"),
            "split_role": case.get("split_role", "test"),
            "bin_time": case.get("bin_time"),
            "ground_truth": case.get("ground_truth") if include_ground_truth else None,
            "prediction_mode": case.get("prediction_mode", default_prediction_mode),
        },
    )


def select_case_predictions(
    case: dict[str, Any],
    *,
    allowed_origins: set[str] | None = None,
    origin: str | None = None,
    predictor_name: str | None = None,
) -> pd.DataFrame:
    rows = pd.DataFrame(case.get("predictions", []))
    if rows.empty:
        return rows

    if allowed_origins is not None:
        if "origin" not in rows.columns:
            return rows.iloc[0:0].copy()
        normalized_origins = {str(item) for item in allowed_origins}
        rows = rows[rows["origin"].astype(str).isin(sorted(normalized_origins))].copy()
    if not rows.empty and origin is not None:
        if "origin" not in rows.columns:
            return rows.iloc[0:0].copy()
        rows = rows[rows["origin"].astype(str) == str(origin)].copy()
    if not rows.empty and predictor_name is not None:
        if "predictor_name" not in rows.columns:
            return rows.iloc[0:0].copy()
        rows = rows[rows["predictor_name"].astype(str) == str(predictor_name)].copy()
    return rows.reset_index(drop=True)
