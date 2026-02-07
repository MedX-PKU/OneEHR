import pandas as pd
import pytest

from oneehr.data.static_postprocess import _encode_static_categoricals, fit_transform_static_features


def test_static_postprocess_onehot_and_impute():
    raw_train = pd.DataFrame(
        {
            "age": [10.0, None],
            "sex": ["M", "F"],
        },
        index=["p1", "p2"],
    )
    raw_val = pd.DataFrame({"age": [None], "sex": ["M"]}, index=["p3"])

    X_tr, X_va, _, art = fit_transform_static_features(
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=None,
        pipeline=[{"op": "impute", "cols": "num__*", "strategy": "mean"}],
    )

    assert "num__age" in X_tr.columns
    assert any(c.startswith("cat__sex__") for c in X_tr.columns)
    assert X_tr.isna().sum().sum() == 0
    assert X_va is not None
    assert X_va.isna().sum().sum() == 0
    assert art.feature_columns == list(X_tr.columns)


def test_static_patient_id_not_a_feature():
    raw = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "age": [10.0, 20.0],
        }
    )
    X = _encode_static_categoricals(raw)
    assert all("patient_id" not in c for c in X.columns)
