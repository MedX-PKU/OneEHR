"""Tests for newly added DL models."""

from __future__ import annotations

import pytest
import torch


B, T, INPUT_DIM, OUT_DIM = 2, 5, 4, 1
HIDDEN = 16


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_cnn(mode, expected_shape):
    from oneehr.models.cnn import CNNPatientModel, CNNTimeModel

    cls = CNNTimeModel if mode == "time" else CNNPatientModel
    m = cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM, num_layers=2)
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    out.sum().backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_grud(mode, expected_shape):
    from oneehr.models.grud import GRUDModel, GRUDTimeModel

    cls = GRUDTimeModel if mode == "time" else GRUDModel
    m = cls(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        feature_means=torch.zeros(INPUT_DIM),
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    missing_mask = torch.zeros(B, T, INPUT_DIM)
    missing_mask[:, 1:, 0] = 1.0
    time_delta = torch.rand(B, T, INPUT_DIM)
    out = m(x, lengths, missing_mask=missing_mask, time_delta=time_delta)
    assert out.shape == expected_shape
    out.sum().backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_sand(mode, expected_shape):
    from oneehr.models.sand import SAnDModel, SAnDTimeModel

    cls = SAnDTimeModel if mode == "time" else SAnDModel
    m = cls(
        input_dim=INPUT_DIM,
        d_model=HIDDEN,
        out_dim=OUT_DIM,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        interp_points=4,
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    out.sum().backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
@pytest.mark.parametrize("attention_type", ["location", "general", "concat"])
def test_dipole(mode, expected_shape, attention_type):
    from oneehr.models.dipole import DipoleModel, DipoleTimeModel

    cls = DipoleTimeModel if mode == "time" else DipoleModel
    m = cls(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        attention_type=attention_type,
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    out.sum().backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_hitanet(mode, expected_shape):
    from oneehr.models.hitanet import HiTANetModel, HiTANetTimeModel

    cls = HiTANetTimeModel if mode == "time" else HiTANetModel
    m = cls(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        group_indices=[[0, 1], [2], [3]],
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    visit_time = torch.arange(T, dtype=torch.float32).repeat(B, 1)
    time_delta = torch.rand(B, T, INPUT_DIM)
    out = m(x, lengths, visit_time=visit_time, time_delta=time_delta)
    assert out.shape == expected_shape
    out.sum().backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_lsan(mode, expected_shape):
    from oneehr.models.lsan import LSANModel, LSANTimeModel

    cls = LSANTimeModel if mode == "time" else LSANModel
    m = cls(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        nhead=4,
        num_layers=1,
        group_indices=[[0, 1], [2], [3]],
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    out.sum().backward()


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


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_m3care(mode, expected_shape):
    from oneehr.models.m3care import M3CareModel, M3CareTimeModel

    cls = M3CareTimeModel if mode == "time" else M3CareModel
    m = cls(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        num_heads=4,
        dim_feedforward=32,
        num_layers=1,
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    out = m(x, lengths)
    assert out.shape == expected_shape
    out.sum().backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
@pytest.mark.parametrize("use_static", [False, True])
def test_safari(mode, expected_shape, use_static):
    from oneehr.models.safari import SafariModel, SafariTimeModel

    cls = SafariTimeModel if mode == "time" else SafariModel
    kwargs = {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN,
        "out_dim": OUT_DIM,
        "dim_list": [1] * INPUT_DIM,
    }
    if use_static:
        kwargs["static_dim"] = 3
    m = cls(**kwargs)
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    static = torch.randn(B, 3) if use_static else None
    out = m(x, lengths, static=static)
    assert out.shape == expected_shape
    out.sum().backward()


@pytest.mark.parametrize("mode,expected_shape", [
    ("patient", (B, OUT_DIM)),
    ("time", (B, T, OUT_DIM)),
])
def test_pai(mode, expected_shape):
    from oneehr.models.pai import PAIModel, PAITimeModel

    cls = PAITimeModel if mode == "time" else PAIModel
    m = cls(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        prompt_init_values=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    missing_mask = torch.zeros(B, T, INPUT_DIM)
    missing_mask[:, 1:, 0] = 1.0
    out = m(x, lengths, missing_mask=missing_mask)
    assert out.shape == expected_shape
    out.sum().backward()


def test_pai_prompt_only_replaces_true_missing_positions():
    from oneehr.models.pai import PAIModel

    m = PAIModel(
        input_dim=2,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        prompt_init_values=torch.tensor([5.0, 7.0]),
    )
    x = torch.tensor([[[0.0, 0.0], [1.0, 2.0]]])
    missing_mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    prompted = m.encoder.apply_prompt(x, missing_mask)
    assert prompted[0, 0, 0].item() == pytest.approx(5.0)
    assert prompted[0, 0, 1].item() == pytest.approx(0.0)
    assert prompted[0, 1, 0].item() == pytest.approx(1.0)
    assert prompted[0, 1, 1].item() == pytest.approx(7.0)
