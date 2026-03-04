"""Tests for static branches (ConCare, GRASP) and TCN patient-level support."""

from __future__ import annotations

import torch

from oneehr.data.features import has_static_branch


# -- ConCare -----------------------------------------------------------------

def test_concare_patient_no_static():
    from oneehr.models.concare import ConCareModel

    m = ConCareModel(input_dim=4, hidden_dim=16, num_heads=4)
    x = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    out = m(x, lengths)
    assert out.shape == (2, 1)


def test_concare_patient_with_static():
    from oneehr.models.concare import ConCareModel

    m = ConCareModel(input_dim=4, hidden_dim=16, num_heads=4, static_dim=3)
    x = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    static = torch.randn(2, 3)
    out = m(x, lengths, static=static)
    assert out.shape == (2, 1)


def test_concare_time_no_static():
    from oneehr.models.concare import ConCareTimeModel

    m = ConCareTimeModel(input_dim=4, hidden_dim=16, num_heads=4)
    x = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    out = m(x, lengths)
    assert out.shape == (2, 5, 1)


def test_concare_time_with_static():
    from oneehr.models.concare import ConCareTimeModel

    m = ConCareTimeModel(input_dim=4, hidden_dim=16, num_heads=4, static_dim=3)
    x = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    static = torch.randn(2, 3)
    out = m(x, lengths, static=static)
    assert out.shape == (2, 5, 1)


# -- GRASP -------------------------------------------------------------------

def test_grasp_patient_no_static():
    from oneehr.models.grasp import GRASPModel

    m = GRASPModel(input_dim=4, hidden_dim=16, cluster_num=2)
    x = torch.randn(3, 5, 4)
    lengths = torch.tensor([5, 3, 4])
    out = m(x, lengths)
    assert out.shape == (3, 1)


def test_grasp_patient_with_static():
    from oneehr.models.grasp import GRASPModel

    m = GRASPModel(input_dim=4, hidden_dim=16, cluster_num=2, static_dim=3)
    x = torch.randn(3, 5, 4)
    lengths = torch.tensor([5, 3, 4])
    static = torch.randn(3, 3)
    out = m(x, lengths, static=static)
    assert out.shape == (3, 1)


def test_grasp_time_no_static():
    from oneehr.models.grasp import GRASPTimeModel

    m = GRASPTimeModel(input_dim=4, hidden_dim=16, cluster_num=2)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([3, 2])
    out = m(x, lengths)
    assert out.shape == (2, 3, 1)


def test_grasp_time_with_static():
    from oneehr.models.grasp import GRASPTimeModel

    m = GRASPTimeModel(input_dim=4, hidden_dim=16, cluster_num=2, static_dim=3)
    x = torch.randn(2, 3, 4)
    lengths = torch.tensor([3, 2])
    static = torch.randn(2, 3)
    out = m(x, lengths, static=static)
    assert out.shape == (2, 3, 1)


# -- TCN patient-level -------------------------------------------------------

def test_tcn_patient_forward():
    from oneehr.models.tcn import TCNPatientModel

    m = TCNPatientModel(input_dim=4, hidden_dim=16, out_dim=1)
    x = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    out = m(x, lengths)
    assert out.shape == (2, 1)


def test_tcn_patient_accepts_static_kwarg():
    from oneehr.models.tcn import TCNPatientModel

    m = TCNPatientModel(input_dim=4, hidden_dim=16, out_dim=1)
    x = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    out = m(x, lengths, static=torch.randn(2, 3))
    assert out.shape == (2, 1)


# -- has_static_branch -------------------------------------------------------

def test_has_static_branch_true():
    from oneehr.models.concare import ConCareModel
    from oneehr.models.grasp import GRASPModel
    from oneehr.models.dragent import DrAgentModel

    assert has_static_branch(ConCareModel(input_dim=4, hidden_dim=16, num_heads=4, static_dim=3))
    assert has_static_branch(GRASPModel(input_dim=4, hidden_dim=16, cluster_num=2, static_dim=3))
    assert has_static_branch(DrAgentModel(input_dim=4, hidden_dim=16, static_dim=3))


def test_has_static_branch_false_when_no_static_dim():
    from oneehr.models.concare import ConCareModel
    from oneehr.models.grasp import GRASPModel

    assert not has_static_branch(ConCareModel(input_dim=4, hidden_dim=16, num_heads=4))
    assert not has_static_branch(GRASPModel(input_dim=4, hidden_dim=16, cluster_num=2))
