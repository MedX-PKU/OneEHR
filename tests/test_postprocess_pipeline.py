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

