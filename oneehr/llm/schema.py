from __future__ import annotations

import json
from typing import Any

from oneehr.llm.contracts import ParsedPrediction


def build_prediction_json_schema(
    *,
    task_kind: str,
    include_explanation: bool,
    include_confidence: bool,
) -> dict[str, Any]:
    if task_kind == "binary":
        props: dict[str, Any] = {
            "label": {"type": "integer", "enum": [0, 1]},
            "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        }
        required = ["label", "probability"]
    elif task_kind == "regression":
        props = {"value": {"type": "number"}}
        required = ["value"]
    else:
        raise ValueError(f"Unsupported llm task kind: {task_kind!r}")

    if include_explanation:
        props["explanation"] = {"type": "string"}
    if include_confidence:
        props["confidence"] = {"type": "number", "minimum": 0.0, "maximum": 1.0}

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": required,
    }


def build_response_format(
    *,
    task_kind: str,
    include_explanation: bool,
    include_confidence: bool,
    supports_json_schema: bool,
) -> dict[str, Any]:
    schema = build_prediction_json_schema(
        task_kind=task_kind,
        include_explanation=include_explanation,
        include_confidence=include_confidence,
    )
    if supports_json_schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "oneehr_prediction",
                "strict": True,
                "schema": schema,
            },
        }
    return {"type": "json_object"}


def schema_prompt_text(
    *,
    task_kind: str,
    include_explanation: bool,
    include_confidence: bool,
) -> str:
    schema = build_prediction_json_schema(
        task_kind=task_kind,
        include_explanation=include_explanation,
        include_confidence=include_confidence,
    )
    return json.dumps(schema, indent=2, sort_keys=True)


def parse_prediction_response(
    text: str,
    *,
    task_kind: str,
    include_explanation: bool,
    include_confidence: bool,
) -> ParsedPrediction:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        return ParsedPrediction(
            parsed_ok=False,
            prediction=None,
            error_code="invalid_json",
            error_message="response did not contain a JSON object",
        )

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParsedPrediction(
            parsed_ok=False,
            prediction=None,
            error_code="invalid_json",
            error_message=str(exc),
        )

    if not isinstance(payload, dict):
        return ParsedPrediction(
            parsed_ok=False,
            prediction=None,
            error_code="invalid_payload",
            error_message="JSON response must be an object",
        )

    explanation = _coerce_optional_string(payload.get("explanation")) if include_explanation else None
    confidence = _coerce_probability(payload.get("confidence")) if include_confidence else None
    if include_confidence and payload.get("confidence") is not None and confidence is None:
        return ParsedPrediction(
            parsed_ok=False,
            prediction=None,
            error_code="invalid_confidence",
            error_message="confidence must be a number in [0, 1]",
            raw_json=payload,
        )

    if task_kind == "binary":
        label = _coerce_binary_label(payload.get("label"))
        if label is None:
            return ParsedPrediction(
                parsed_ok=False,
                prediction=None,
                error_code="missing_label",
                error_message="label must be 0 or 1",
                raw_json=payload,
            )
        probability = _coerce_probability(payload.get("probability"))
        if probability is None:
            if payload.get("probability") is None:
                probability = float(label)
            else:
                return ParsedPrediction(
                    parsed_ok=False,
                    prediction=None,
                    error_code="invalid_probability",
                    error_message="probability must be a number in [0, 1]",
                    raw_json=payload,
                )
        return ParsedPrediction(
            parsed_ok=True,
            prediction=label,
            probability=probability,
            explanation=explanation,
            confidence=confidence,
            raw_json=payload,
        )

    if task_kind == "regression":
        value = _coerce_float(payload.get("value"))
        if value is None:
            return ParsedPrediction(
                parsed_ok=False,
                prediction=None,
                error_code="missing_value",
                error_message="value must be numeric for regression tasks",
                raw_json=payload,
            )
        return ParsedPrediction(
            parsed_ok=True,
            prediction=value,
            value=value,
            explanation=explanation,
            confidence=confidence,
            raw_json=payload,
        )

    raise ValueError(f"Unsupported llm task kind: {task_kind!r}")


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


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _coerce_probability(value: Any) -> float | None:
    out = _coerce_float(value)
    if out is None:
        return None
    if out < 0.0 or out > 1.0:
        return None
    return out


def _coerce_binary_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if int(value) in {0, 1} and float(value) in {0.0, 1.0}:
            return int(value)
        return None
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in {"0", "false", "negative", "no"}:
        return 0
    if txt in {"1", "true", "positive", "yes"}:
        return 1
    return None

