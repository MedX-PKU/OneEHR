"""Tests for new analysis modules: fairness, calibration, statistical_tests, missing_data."""

import numpy as np
import pandas as pd


def _make_preds(n=200, seed=42):
    """Create synthetic predictions DataFrame."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "system": ["model_a"] * (n // 2) + ["model_b"] * (n - n // 2),
        "y_true": y_true,
        "y_pred": y_pred,
    })


def _make_static(n=200):
    """Create synthetic static features."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "age": rng.integers(20, 90, size=n),
        "sex": rng.choice(["M", "F"], size=n),
    })


# ── Fairness ──

def test_fairness_detects_sensitive_columns():
    from oneehr.analysis.fairness import _detect_sensitive_columns
    df = pd.DataFrame({"age": [1], "sex": [1], "bp": [1], "ethnicity_group": [1]})
    cols = _detect_sensitive_columns(df)
    assert "age" in cols
    assert "sex" in cols
    assert "ethnicity_group" in cols
    assert "bp" not in cols


def test_fairness_compute():
    from oneehr.analysis.fairness import compute_fairness
    preds = _make_preds()
    static = _make_static()
    result = compute_fairness(preds=preds, static=static)
    assert "sensitive_columns" in result
    assert len(result["systems"]) == 2
    for sys in result["systems"]:
        assert "attributes" in sys
        for attr_data in sys["attributes"].values():
            assert "demographic_parity_diff" in attr_data
            assert "equalized_odds_diff" in attr_data
            assert "groups" in attr_data


# ── Calibration ──

def test_calibration_compute():
    from oneehr.analysis.calibration import compute_calibration
    n = 300
    preds = _make_preds(n=n)
    split_info = {
        "val": [f"p{i}" for i in range(0, 100)],
        "test": [f"p{i}" for i in range(100, n)],
    }
    result, cal_df = compute_calibration(preds=preds, split_info=split_info)
    assert result["module"] == "calibration"
    assert len(result["systems"]) > 0
    for sys in result["systems"]:
        assert "methods" in sys
        for method_name, method_data in sys["methods"].items():
            if "error" not in method_data:
                assert "pre_ece" in method_data
                assert "post_ece" in method_data


# ── Statistical Tests ──

def test_statistical_tests_compute():
    from oneehr.analysis.statistical_tests import compute_statistical_tests
    # Both systems must share the same patients for pairwise comparison
    rng = np.random.default_rng(42)
    n = 100
    pids = [f"p{i}" for i in range(n)]
    y_true = rng.integers(0, 2, size=n).astype(float)
    preds = pd.concat([
        pd.DataFrame({"patient_id": pids, "system": "model_a",
                       "y_true": y_true, "y_pred": np.clip(y_true + rng.normal(0, 0.3, n), 0, 1)}),
        pd.DataFrame({"patient_id": pids, "system": "model_b",
                       "y_true": y_true, "y_pred": np.clip(y_true + rng.normal(0, 0.4, n), 0, 1)}),
    ], ignore_index=True)
    result = compute_statistical_tests(preds=preds)
    assert result["module"] == "statistical_tests"
    assert len(result["pairwise"]) == 1  # 2 systems -> 1 pair
    pair = result["pairwise"][0]
    assert "delong" in pair
    assert "mcnemar" in pair
    assert "p_value" in pair["delong"]
    assert "p_value" in pair["mcnemar"]


def test_delong_identical_models():
    from oneehr.analysis.statistical_tests import _delong_roc_test
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100).astype(float)
    y_pred = np.clip(y_true + rng.normal(0, 0.2, size=100), 0, 1)
    z, p = _delong_roc_test(y_true, y_pred, y_pred)
    assert abs(z) < 1e-10
    assert p > 0.99


# ── Missing Data ──

def test_missing_data_compute():
    from oneehr.analysis.missing_data import compute_missing_data
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "patient_id": range(n),
        "bin_time": range(n),
        "num__hr": rng.normal(80, 10, size=n),
        "num__bp": rng.normal(120, 15, size=n),
    })
    # Inject some missing values
    df.loc[df.index[:20], "num__hr"] = np.nan
    df.loc[df.index[:15], "num__bp"] = np.nan

    result = compute_missing_data(binned=df)
    assert result["module"] == "missing_data"
    assert result["summary"]["n_features_with_missing"] == 2
    assert result["features"]["num__hr"]["rate"] > 0
    assert result["features"]["num__bp"]["rate"] > 0


def test_missing_data_no_missing():
    from oneehr.analysis.missing_data import compute_missing_data
    df = pd.DataFrame({
        "patient_id": range(10),
        "bin_time": range(10),
        "num__hr": np.ones(10),
    })
    result = compute_missing_data(binned=df)
    assert result["summary"]["n_features_with_missing"] == 0
