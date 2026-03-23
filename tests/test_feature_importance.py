import numpy as np


def test_attention_importance_shapes():
    from oneehr.analysis.feature_importance import attention_importance

    X = np.ones((3, 5, 4), dtype=float)
    attn = np.zeros((3, 5), dtype=float)
    attn[:, -1] = 1.0
    res = attention_importance(attn, X, feature_names=["a", "b", "c", "d"])
    assert res.importances.shape == (4,)
    assert res.feature_names == ["a", "b", "c", "d"]


def test_as_last_step_for_sequence_tabular():
    from oneehr.analysis.feature_importance import _as_2d_tabular_input

    X = np.zeros((2, 3, 2), dtype=float)
    X[:, -1, :] = np.array([[1.0, 2.0], [3.0, 4.0]])
    x2d, feats, kind = _as_2d_tabular_input(X, feature_names=["f1", "f2"])
    assert kind == "sequence"
    assert feats == ["f1", "f2"]
    assert x2d.shape == (2, 2)
    assert np.allclose(x2d, np.array([[1.0, 2.0], [3.0, 4.0]]))
