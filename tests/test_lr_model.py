"""Tests for logistic regression baseline model."""

import numpy as np
import pandas as pd


def test_lr_in_tabular_models():
    from oneehr.models import TABULAR_MODELS

    assert "lr" in TABULAR_MODELS


def test_lr_binary_train_predict():
    from oneehr.config.schema import TaskConfig
    from oneehr.models.tree import predict_tabular, train_tabular_model

    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n).astype(float)
    task = TaskConfig(kind="binary", prediction_mode="patient")

    art = train_tabular_model(
        model_name="lr",
        X_train=X,
        y_train=y,
        X_val=None,
        y_val=None,
        task=task,
        params={},
    )
    assert art.kind == "lr"
    preds = predict_tabular(art, X, task)
    assert preds.shape == (n,)
    assert np.all((preds >= 0) & (preds <= 1))


def test_lr_regression_train_predict():
    from oneehr.config.schema import TaskConfig
    from oneehr.models.tree import predict_tabular, train_tabular_model

    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = rng.normal(size=n)
    task = TaskConfig(kind="regression", prediction_mode="patient")

    art = train_tabular_model(
        model_name="lr",
        X_train=X,
        y_train=y,
        X_val=None,
        y_val=None,
        task=task,
        params={},
    )
    assert art.kind == "lr"
    preds = predict_tabular(art, X, task)
    assert preds.shape == (n,)


def test_lr_save_load(tmp_path):
    from oneehr.config.schema import TaskConfig
    from oneehr.models.tree import (
        load_tabular_model,
        predict_tabular,
        save_tabular_model,
        train_tabular_model,
    )

    rng = np.random.default_rng(42)
    n = 50
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n).astype(float)
    task = TaskConfig(kind="binary", prediction_mode="patient")

    art = train_tabular_model(
        model_name="lr",
        X_train=X,
        y_train=y,
        X_val=None,
        y_val=None,
        task=task,
        params={},
    )
    save_tabular_model(art, tmp_path / "lr_model")
    loaded = load_tabular_model(tmp_path / "lr_model", task=task, kind="lr")
    preds_orig = predict_tabular(art, X, task)
    preds_loaded = predict_tabular(loaded, X, task)
    np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
