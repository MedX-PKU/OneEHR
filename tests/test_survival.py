"""Tests for survival analysis models and metrics."""

import numpy as np
import pytest
import torch

B, T, INPUT_DIM, OUT_DIM = 4, 5, 8, 1
HIDDEN = 16


def test_deepsurv_forward():
    from oneehr.models.survival import DeepSurv

    m = DeepSurv(input_dim=INPUT_DIM, hidden_dim=HIDDEN, num_layers=2, out_dim=OUT_DIM)
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 1, T - 2, T])
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)
    out.sum().backward()


def test_deepsurv_2d_input():
    from oneehr.models.survival import DeepSurv

    m = DeepSurv(input_dim=INPUT_DIM, hidden_dim=HIDDEN, out_dim=OUT_DIM)
    x = torch.randn(B, INPUT_DIM)
    lengths = torch.ones(B)
    out = m(x, lengths)
    assert out.shape == (B, OUT_DIM)


def test_deephit_forward():
    from oneehr.models.survival import DeepHit

    num_bins = 10
    m = DeepHit(input_dim=INPUT_DIM, hidden_dim=HIDDEN, num_time_bins=num_bins)
    x = torch.randn(B, T, INPUT_DIM)
    lengths = torch.tensor([T, T - 1, T - 2, T])
    out = m(x, lengths)
    assert out.shape == (B, num_bins)
    # Output should be valid probability distribution
    assert torch.allclose(out.sum(dim=1), torch.ones(B), atol=1e-5)
    out.sum().backward()


def test_cox_ph_loss():
    from oneehr.models.survival import CoxPHLoss

    loss_fn = CoxPHLoss()
    risk = torch.randn(10, requires_grad=True)
    times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    events = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    loss = loss_fn(risk, times, events)
    assert loss.ndim == 0
    loss.backward()


def test_concordance_index_perfect():
    from oneehr.eval.survival import concordance_index

    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # higher risk = shorter time
    events = np.ones(5)
    ci = concordance_index(times, risk, events)
    assert ci == pytest.approx(1.0)


def test_concordance_index_random():
    from oneehr.eval.survival import concordance_index

    rng = np.random.default_rng(42)
    times = rng.uniform(0, 10, size=50)
    risk = rng.uniform(0, 1, size=50)
    events = np.ones(50)
    ci = concordance_index(times, risk, events)
    # Random predictions should give C-index around 0.5
    assert 0.3 < ci < 0.7


def test_survival_metrics():
    from oneehr.eval.survival import survival_metrics

    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    events = np.array([1, 1, 0, 1, 0])

    result = survival_metrics(times, risk, events)
    assert "c_index" in result.metrics
    assert result.metrics["n_events"] == 3
    assert result.metrics["n_censored"] == 2


def test_brier_score_at_time():
    from oneehr.eval.survival import brier_score_at_time

    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.ones(5)
    # Perfect predictions: S(t=3) = 0 for those who died before t=3
    pred_surv = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    bs = brier_score_at_time(times, events, pred_surv, t=3.0)
    assert 0 <= bs <= 1
