"""Tests for new preprocessing ops: knn_impute, iterative_impute, robust_scale, quantile_norm."""

import numpy as np
import pandas as pd


def _make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "num__hr": rng.normal(80, 10, size=n),
        "num__bp": rng.normal(120, 15, size=n),
    })
    # Inject missing values
    df.loc[df.index[:15], "num__hr"] = np.nan
    df.loc[df.index[:10], "num__bp"] = np.nan
    return df


def test_knn_impute():
    from oneehr.data.tabular import fit_postprocess_pipeline, transform_postprocess_pipeline
    df = _make_df()
    pipeline = [{"op": "knn_impute", "cols": "num__*", "n_neighbors": 3}]
    fitted = fit_postprocess_pipeline(df, pipeline)
    out = transform_postprocess_pipeline(df, fitted)
    assert out.isna().sum().sum() == 0


def test_iterative_impute():
    from oneehr.data.tabular import fit_postprocess_pipeline, transform_postprocess_pipeline
    df = _make_df()
    pipeline = [{"op": "iterative_impute", "cols": "num__*", "max_iter": 5}]
    fitted = fit_postprocess_pipeline(df, pipeline)
    out = transform_postprocess_pipeline(df, fitted)
    assert out.isna().sum().sum() == 0


def test_robust_scale():
    from oneehr.data.tabular import fit_postprocess_pipeline, transform_postprocess_pipeline
    df = _make_df().fillna(0)
    pipeline = [{"op": "robust_scale", "cols": "num__*"}]
    fitted = fit_postprocess_pipeline(df, pipeline)
    out = transform_postprocess_pipeline(df, fitted)
    # Scaled data should have median near 0 for training data
    assert abs(out["num__hr"].median()) < 1.0
    assert abs(out["num__bp"].median()) < 1.0


def test_quantile_norm():
    from oneehr.data.tabular import fit_postprocess_pipeline, transform_postprocess_pipeline
    df = _make_df().fillna(0)
    pipeline = [{"op": "quantile_norm", "cols": "num__*", "output_distribution": "normal"}]
    fitted = fit_postprocess_pipeline(df, pipeline)
    out = transform_postprocess_pipeline(df, fitted)
    # Quantile-normalized data should have values roughly in [-6, 6] for normal output
    assert out["num__hr"].max() < 10.0
    assert out["num__hr"].min() > -10.0


def test_fit_on_train_transform_test():
    from oneehr.data.tabular import maybe_fit_transform_postprocess
    train = _make_df(n=80, seed=0)
    test = _make_df(n=20, seed=1)
    pipeline = [
        {"op": "knn_impute", "cols": "num__*", "n_neighbors": 3},
        {"op": "robust_scale", "cols": "num__*"},
    ]
    train_t, _, test_t, fitted = maybe_fit_transform_postprocess(train, None, test, pipeline)
    assert train_t.isna().sum().sum() == 0
    assert test_t.isna().sum().sum() == 0
    assert fitted is not None
