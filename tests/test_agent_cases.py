from __future__ import annotations

from oneehr.agent.cases import build_case_context, case_static_features, select_case_predictions


def test_build_case_context_materializes_prompt_inputs() -> None:
    case = {
        "case_id": "case-001",
        "patient_id": "p0001",
        "split": "fold0",
        "split_role": "test",
        "bin_time": "2020-01-02T00:00:00",
        "ground_truth": 1,
        "prediction_mode": "time",
        "events": [
            {"event_time": "2020-01-01T00:00:00", "code": "LAB_A", "value": 1.2},
            {"event_time": "2020-01-02T00:00:00", "code": "LAB_B", "value": 3.4},
        ],
        "static": {"features": {"age": 64, "sex": "F"}},
    }

    context = build_case_context(case, default_prediction_mode="patient", include_ground_truth=False)

    assert context.case_id == "case-001"
    assert context.patient_id == "p0001"
    assert list(context.dynamic["code"]) == ["LAB_A", "LAB_B"]
    assert context.static_row is not None
    assert context.static_row["age"] == 64
    assert context.instance["instance_id"] == "case-001"
    assert context.instance["ground_truth"] is None
    assert context.instance["prediction_mode"] == "time"


def test_case_static_features_handles_missing_payloads() -> None:
    assert case_static_features({"static": None}) == {}
    assert case_static_features({"static": {"features": {"age": 55}}}) == {"age": 55}


def test_select_case_predictions_supports_origin_and_predictor_filters() -> None:
    case = {
        "predictions": [
            {"origin": "model", "predictor_name": "xgboost", "score": 0.9},
            {"origin": "agent", "predictor_name": "gpt-review", "score": 0.8},
            {"origin": "model", "predictor_name": "rf", "score": 0.7},
        ]
    }

    model_rows = select_case_predictions(case, allowed_origins={"model"})
    assert model_rows["predictor_name"].tolist() == ["xgboost", "rf"]

    target_rows = select_case_predictions(case, origin="model", predictor_name="xgboost")
    assert len(target_rows) == 1
    assert target_rows.iloc[0]["score"] == 0.9
