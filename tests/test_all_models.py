"""Tests for all DL model architectures — forward pass + gradient flow."""

import pytest
import torch

B, T, INPUT_DIM, OUT_DIM = 2, 5, 4, 1
HIDDEN = 16
STATIC_DIM = 3


def _basic_inputs():
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 2])
    return x, lengths


def _static_input():
    return torch.randn(B, STATIC_DIM)


@pytest.mark.parametrize("cell", ["gru", "lstm", "rnn"])
def test_recurrent_patient(cell):
    from oneehr.models.recurrent import RecurrentModel

    m = RecurrentModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM, cell=cell)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


@pytest.mark.parametrize("cell", ["gru", "lstm", "rnn"])
def test_recurrent_time(cell):
    from oneehr.models.recurrent import RecurrentTimeModel

    m = RecurrentTimeModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM, cell=cell)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, T, OUT_DIM)
    out.sum().backward()


def test_transformer_patient():
    from oneehr.models.transformer import TransformerModel

    m = TransformerModel(input_dim=INPUT_DIM, d_model=HIDDEN, out_dim=OUT_DIM, nhead=2, num_layers=1)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_transformer_time():
    from oneehr.models.transformer import TransformerTimeModel

    m = TransformerTimeModel(input_dim=INPUT_DIM, d_model=HIDDEN, out_dim=OUT_DIM, nhead=2, num_layers=1)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, T, OUT_DIM)
    out.sum().backward()


def test_tcn_patient():
    from oneehr.models.tcn import TCNPatientModel

    m = TCNPatientModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM, num_layers=2, kernel_size=3)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_mlp_patient():
    from oneehr.models.mlp import MLPModel

    m = MLPModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_adacare_patient():
    from oneehr.models.adacare import AdaCareModel

    m = AdaCareModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM, kernel_size=2, kernel_num=8)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_stagenet_patient():
    from oneehr.models.stagenet import StageNetModel

    m = StageNetModel(input_dim=INPUT_DIM, chunk_size=HIDDEN, levels=2, conv_size=3, out_dim=OUT_DIM)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_retain_patient():
    from oneehr.models.retain import RETAINModel

    m = RETAINModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


@pytest.mark.parametrize("use_static", [False, True])
def test_concare_patient(use_static):
    from oneehr.models.concare import ConCareModel

    sd = STATIC_DIM if use_static else 0
    m = ConCareModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, num_heads=2, out_dim=OUT_DIM, static_dim=sd)
    x, lengths = _basic_inputs()
    s = _static_input() if use_static else None
    out = m(x, lengths, static=s)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


@pytest.mark.parametrize("use_static", [False, True])
def test_grasp_patient(use_static):
    from oneehr.models.grasp import GRASPModel

    sd = STATIC_DIM if use_static else 0
    m = GRASPModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, cluster_num=4, out_dim=OUT_DIM, static_dim=sd)
    x, lengths = _basic_inputs()
    s = _static_input() if use_static else None
    out = m(x, lengths, static=s)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_mcgru_patient():
    """MCGRU requires static_dim > 0."""
    from oneehr.models.mcgru import MCGRUModel

    m = MCGRUModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN, feat_dim=4, out_dim=OUT_DIM, static_dim=STATIC_DIM)
    x, lengths = _basic_inputs()
    s = _static_input()
    out = m(x, lengths, static=s)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


@pytest.mark.parametrize("use_static", [False, True])
def test_dragent_patient(use_static):
    from oneehr.models.dragent import DrAgentModel

    sd = STATIC_DIM if use_static else 0
    m = DrAgentModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN,
        out_dim=OUT_DIM,
        static_dim=sd,
        n_actions=4,
        n_units=8,
    )
    x, lengths = _basic_inputs()
    s = _static_input() if use_static else None
    out = m(x, lengths, static=s)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_deepsurv():
    from oneehr.models.survival import DeepSurv

    m = DeepSurv(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_deephit():
    from oneehr.models.survival import DeepHit

    m = DeepHit(input_dim=INPUT_DIM, hidden_dim=HIDDEN, num_time_bins=10)
    x, lengths = _basic_inputs()
    out = m(x, lengths)
    assert out.shape == (B, 10)
    assert torch.allclose(out.sum(dim=1), torch.ones(B), atol=1e-5)
    out.sum().backward()


def test_model_registry_build():
    """Test that build_dl_model works for all registered models."""
    from oneehr.config.schema import ModelConfig
    from oneehr.models import DL_MODELS, build_dl_model

    skip = {"deephit", "mcgru"}  # DeepHit has different output dim, MCGRU requires static
    for name in sorted(DL_MODELS - skip):
        cfg = ModelConfig(name=name, params={})
        try:
            model = build_dl_model(cfg, input_dim=INPUT_DIM, out_dim=OUT_DIM, mode="patient")
            assert model is not None, f"build_dl_model returned None for {name}"
        except Exception as e:
            pytest.fail(f"build_dl_model failed for {name}: {e}")
