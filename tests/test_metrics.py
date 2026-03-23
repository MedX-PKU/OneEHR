"""Tests for evaluation metrics."""

import numpy as np
import pytest


def test_binary_metrics_basic():
    from oneehr.eval.metrics import binary_metrics

    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.6, 0.4])

    result = binary_metrics(y_true, y_score)
    m = result.metrics
    assert 0 <= m["auroc"] <= 1
    assert 0 <= m["auprc"] <= 1
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1"] <= 1
    assert -1 <= m["mcc"] <= 1
    assert 0 <= m["sensitivity"] <= 1
    assert 0 <= m["specificity"] <= 1
    assert m["brier"] >= 0
    assert m["ece"] >= 0


def test_binary_metrics_perfect():
    from oneehr.eval.metrics import binary_metrics

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.0, 1.0, 1.0])
    result = binary_metrics(y_true, y_score)
    assert result.metrics["auroc"] == pytest.approx(1.0)
    assert result.metrics["accuracy"] == pytest.approx(1.0)


def test_binary_metrics_single_class():
    from oneehr.eval.metrics import binary_metrics

    y_true = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7])
    result = binary_metrics(y_true, y_score)
    assert np.isnan(result.metrics["auroc"])  # undefined for single class


def test_multiclass_metrics():
    from oneehr.eval.metrics import multiclass_metrics

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=100)
    y_score = rng.dirichlet([1, 1, 1], size=100)

    result = multiclass_metrics(y_true, y_score, num_classes=3)
    m = result.metrics
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1_macro"] <= 1
    assert 0 <= m["f1_micro"] <= 1


def test_regression_metrics():
    from oneehr.eval.metrics import regression_metrics

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

    result = regression_metrics(y_true, y_pred)
    m = result.metrics
    assert m["mae"] >= 0
    assert m["rmse"] >= 0
    assert m["r2"] <= 1


def test_multilabel_metrics():
    from oneehr.eval.metrics import multilabel_metrics

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=(50, 5))
    y_score = rng.random(size=(50, 5))

    result = multilabel_metrics(y_true, y_score)
    m = result.metrics
    assert 0 <= m["subset_accuracy"] <= 1
    assert 0 <= m["hamming_loss"] <= 1
    assert 0 <= m["f1_macro"] <= 1
    assert 0 <= m["f1_micro"] <= 1


def test_net_benefit():
    from oneehr.eval.metrics import net_benefit

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100).astype(float)
    y_score = np.clip(y_true + rng.normal(0, 0.3, size=100), 0, 1)

    result = net_benefit(y_true, y_score)
    assert "thresholds" in result
    assert "net_benefit_model" in result
    assert "net_benefit_treat_all" in result
    assert len(result["thresholds"]) == len(result["net_benefit_model"])


def test_sensitivity_specificity_at_thresholds():
    from oneehr.eval.metrics import sensitivity_specificity_at_thresholds

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])

    result = sensitivity_specificity_at_thresholds(y_true, y_score, thresholds=[0.3, 0.5, 0.7])
    assert "t=0.50" in result
    assert 0 <= result["t=0.50"]["sensitivity"] <= 1


def test_bootstrap_ci():
    from oneehr.config.schema import TaskConfig
    from oneehr.eval.bootstrap import bootstrap_metric

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100).astype(float)
    y_pred = np.clip(y_true + rng.normal(0, 0.3, size=100), 0, 1)

    result = bootstrap_metric(
        y_true=y_true,
        y_pred=y_pred,
        task=TaskConfig(kind="binary"),
        metric="auroc",
        n=50,
        seed=42,
    )
    assert result.ci_low < result.mean < result.ci_high
    assert result.n == 50
