from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedReview:
    parsed_ok: bool
    supported: bool | None = None
    clinically_grounded: bool | None = None
    leakage_suspected: bool | None = None
    needs_human_review: bool | None = None
    overall_score: float | None = None
    review_summary: str | None = None
    key_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    raw_json: dict[str, Any] | None = None


def build_review_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "supported": {"type": "boolean"},
            "clinically_grounded": {"type": "boolean"},
            "leakage_suspected": {"type": "boolean"},
            "needs_human_review": {"type": "boolean"},
            "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "review_summary": {"type": "string"},
            "key_evidence": {"type": "array", "items": {"type": "string"}},
            "missing_evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "supported",
            "clinically_grounded",
            "leakage_suspected",
            "needs_human_review",
            "overall_score",
            "review_summary",
        ],
    }


def build_review_response_format(*, supports_json_schema: bool) -> dict[str, Any]:
    schema = build_review_json_schema()
    if supports_json_schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "oneehr_review",
                "strict": True,
                "schema": schema,
            },
        }
    return {"type": "json_object"}


def review_schema_prompt_text() -> str:
    return json.dumps(build_review_json_schema(), indent=2, sort_keys=True)


def parse_review_response(text: str) -> ParsedReview:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return ParsedReview(
            parsed_ok=False,
            error_code="invalid_json",
            error_message="response did not contain a JSON object",
        )
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParsedReview(
            parsed_ok=False,
            error_code="invalid_json",
            error_message=str(exc),
        )
    if not isinstance(payload, dict):
        return ParsedReview(
            parsed_ok=False,
            error_code="invalid_payload",
            error_message="JSON response must be an object",
        )

    supported = _coerce_bool(payload.get("supported"))
    clinically_grounded = _coerce_bool(payload.get("clinically_grounded"))
    leakage_suspected = _coerce_bool(payload.get("leakage_suspected"))
    needs_human_review = _coerce_bool(payload.get("needs_human_review"))
    overall_score = _coerce_probability(payload.get("overall_score"))
    review_summary = _coerce_optional_string(payload.get("review_summary"))
    key_evidence = _coerce_string_list(payload.get("key_evidence"))
    missing_evidence = _coerce_string_list(payload.get("missing_evidence"))

    missing = []
    for key, value in (
        ("supported", supported),
        ("clinically_grounded", clinically_grounded),
        ("leakage_suspected", leakage_suspected),
        ("needs_human_review", needs_human_review),
        ("overall_score", overall_score),
        ("review_summary", review_summary),
    ):
        if value is None:
            missing.append(key)
    if missing:
        return ParsedReview(
            parsed_ok=False,
            error_code="invalid_review_fields",
            error_message=f"missing or invalid fields: {missing}",
            raw_json=payload,
        )

    return ParsedReview(
        parsed_ok=True,
        supported=supported,
        clinically_grounded=clinically_grounded,
        leakage_suspected=leakage_suspected,
        needs_human_review=needs_human_review,
        overall_score=overall_score,
        review_summary=review_summary,
        key_evidence=key_evidence,
        missing_evidence=missing_evidence,
        raw_json=payload,
    )


def _extract_json_candidate(text: str) -> str | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1, 0.0, 1.0}:
        return bool(value)
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in {"true", "1", "yes"}:
        return True
    if txt in {"false", "0", "no"}:
        return False
    return None


def _coerce_probability(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out < 0.0 or out > 1.0:
        return None
    return out


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            txt = _coerce_optional_string(item)
            if txt:
                out.append(txt)
        return out
    txt = _coerce_optional_string(value)
    return [] if txt is None else [txt]
