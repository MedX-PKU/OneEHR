"""Tests for Deepr, EHR-Mamba, and Jamba models."""

from __future__ import annotations

import pytest
import torch


B, T, INPUT_DIM, OUT_DIM = 2, 5, 4, 1
HIDDEN = 16


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_deepr(mode, expected_shape):
    from oneehr.models.deepr import DeeprModel, DeeprTimeModel

    cls = DeeprTimeModel if mode == "time" else DeeprModel
    m = cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM)
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_mamba(mode, expected_shape):
    from oneehr.models.mamba import EHRMambaModel, EHRMambaTimeModel

    cls = EHRMambaTimeModel if mode == "time" else EHRMambaModel
    m = cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM, num_layers=1, state_size=4)
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_jamba(mode, expected_shape):
    from oneehr.models.jamba import JambaModel, JambaTimeModel

    cls = JambaTimeModel if mode == "time" else JambaModel
    m = cls(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM,
        num_transformer_layers=1, num_mamba_layers=2, heads=2, state_size=4,
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    loss = out.sum()
    loss.backward()
