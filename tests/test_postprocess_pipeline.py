from __future__ import annotations

import numpy as np
import pandas as pd

from oneehr.data.postprocess import fit_postprocess_pipeline, transform_postprocess_pipeline


def test_pipeline_standardize_then_impute_fit_on_train_only() -> None:
    X_train = pd.DataFrame(
        {
            "num__a": [0.0, 1.0, np.nan],
            "num__b": [10.0, 10.0, 10.0],
        }
    )
    X_test = pd.DataFrame({"num__a": [1000.0], "num__b": [10.0]})

    pipeline = [
        {"op": "standardize", "cols": "num__*"},
        {"op": "impute", "strategy": "mean", "cols": "num__*"},
    ]

    fitted = fit_postprocess_pipeline(X_train, pipeline)

    # Ensure fitted stats are from train only.
    std_step = next(s for s in fitted.pipeline if s["op"] == "standardize")
    mean_a = float(std_step["mean"]["num__a"])
    assert np.isclose(mean_a, 0.5)  # mean of [0, 1] ignoring NaN

    Xt = transform_postprocess_pipeline(X_test, fitted)
    # Large test value should not affect mean; it should become huge after transform.
    assert float(Xt.loc[0, "num__a"]) > 100.0


def test_pipeline_winsorize_quantiles() -> None:
    X_train = pd.DataFrame({"num__a": [0.0, 0.0, 0.0, 100.0]})
    pipeline = [{"op": "winsorize", "cols": "num__*", "lower_q": 0.0, "upper_q": 0.75}]
    fitted = fit_postprocess_pipeline(X_train, pipeline)
    Xt = transform_postprocess_pipeline(pd.DataFrame({"num__a": [1000.0]}), fitted)
    assert float(Xt.loc[0, "num__a"]) <= 100.0


def test_pipeline_forward_fill_with_fallback() -> None:
    # Within each patient, forward-fill. If no previous value exists, use fallback.
    X_train = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2"],
            "bin_time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "num__a": [1.0, np.nan, np.nan, 5.0],
            "cat__x": [1.0, np.nan, np.nan, 2.0],
        }
    )
    pipeline = [
        {
            "op": "forward_fill",
            "cols": ["num__a", "cat__x"],
            "group_key": "patient_id",
            "order_key": "bin_time",
            "fallback": {"strategy": "constant", "value": 0.0},
        }
    ]
    fitted = fit_postprocess_pipeline(X_train, pipeline)
    Xt = transform_postprocess_pipeline(X_train, fitted).sort_values(["patient_id", "bin_time"])

    # p1 at 2020-01-02 should carry forward 1.0 and 1.0
    p1_t1 = Xt[(Xt["patient_id"] == "p1") & (Xt["bin_time"] == pd.Timestamp("2020-01-02"))].iloc[0]
    assert float(p1_t1["num__a"]) == 1.0
    assert float(p1_t1["cat__x"]) == 1.0

    # p2 at 2020-01-01 has no previous value => fallback 0.0
    p2_t0 = Xt[(Xt["patient_id"] == "p2") & (Xt["bin_time"] == pd.Timestamp("2020-01-01"))].iloc[0]
    assert float(p2_t0["num__a"]) == 0.0
    assert float(p2_t0["cat__x"]) == 0.0
