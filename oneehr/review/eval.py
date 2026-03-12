from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_review_rows(frame: pd.DataFrame) -> dict[str, Any]:
    total_rows = int(len(frame))
    if total_rows == 0:
        return {
            "total_rows": 0,
            "parsed_ok_rows": 0,
            "parse_success_rate": 0.0,
            "metrics": {},
        }

    parsed = frame[frame["parsed_ok"] == True].copy()  # noqa: E712
    parsed_ok_rows = int(len(parsed))
    metrics: dict[str, float] = {}
    if not parsed.empty:
        for col, key in (
            ("supported", "supported_rate"),
            ("clinically_grounded", "clinically_grounded_rate"),
            ("leakage_suspected", "leakage_suspected_rate"),
            ("needs_human_review", "needs_human_review_rate"),
            ("overall_score", "mean_overall_score"),
        ):
            if col not in parsed.columns:
                continue
            series = pd.to_numeric(parsed[col], errors="coerce")
            if series.notna().any():
                metrics[key] = float(series.mean())
    return {
        "total_rows": total_rows,
        "parsed_ok_rows": parsed_ok_rows,
        "parse_success_rate": float(parsed_ok_rows / total_rows) if total_rows else 0.0,
        "metrics": metrics,
    }
