"""Regression tests for training-loss edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


class _FixedTimeModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, x, lengths):
        return self.logits[: x.shape[0], : x.shape[1]]


def test_dragent_agent_layers_receive_gradients():
    from oneehr.models.dragent import DrAgentModel

    model = DrAgentModel(input_dim=4, hidden_dim=8, out_dim=1, n_actions=4, n_units=6, dropout=0.0)
    x = torch.randn(3, 5, 4)
    lengths = torch.tensor([5, 4, 3])
    y = torch.randn(3, 1)

    loss = (model(x, lengths) - y).pow(2).mean()
    loss.backward()

    agent_grads = [p.grad for name, p in model.named_parameters() if "agent" in name]
    assert agent_grads
    assert all(g is not None and g.abs().sum() > 0 for g in agent_grads)


def test_run_epoch_masks_multilabel_time_loss():
    from oneehr.config.schema import TaskConfig, TrainerConfig
    from oneehr.training.trainer import _run_epoch

    logits = torch.tensor(
        [
            [[0.1, -0.2], [0.5, 0.3], [-0.4, 0.8]],
            [[1.0, -1.0], [0.0, 0.2], [0.7, -0.6]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [1, 0], [0, 0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    model = _FixedTimeModel(logits)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    got = _run_epoch(
        model,
        torch.zeros(2, 3, 4),
        torch.tensor([3, 2]),
        y,
        None,
        mask,
        loss_fn,
        None,
        TrainerConfig(batch_size=2),
        torch.device("cpu"),
        TaskConfig(kind="multilabel", prediction_mode="time", num_classes=2),
        train=False,
    )

    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    expected = (raw * mask.unsqueeze(-1)).sum() / mask.unsqueeze(-1).expand_as(raw).sum()
    assert got == pytest.approx(float(expected))


def test_run_epoch_uses_last_dim_as_time_multiclass_classes():
    from oneehr.config.schema import TaskConfig, TrainerConfig
    from oneehr.training.trainer import _run_epoch

    logits = torch.tensor(
        [
            [[3.0, 0.1, -1.0], [0.0, 2.0, -0.5], [0.2, 0.1, 1.5]],
            [[-0.2, 1.0, 0.0], [2.0, 0.3, -0.1], [0.5, 0.4, 0.3]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    model = _FixedTimeModel(logits)

    got = _run_epoch(
        model,
        torch.zeros(2, 3, 4),
        torch.tensor([3, 2]),
        y.float(),
        None,
        mask,
        torch.nn.CrossEntropyLoss(reduction="none"),
        None,
        TrainerConfig(batch_size=2),
        torch.device("cpu"),
        TaskConfig(kind="multiclass", prediction_mode="time", num_classes=3),
        train=False,
    )

    raw = torch.nn.functional.cross_entropy(logits.movedim(-1, 1), y, reduction="none")
    expected = (raw * mask).sum() / mask.sum()
    assert got == pytest.approx(float(expected))


def test_fit_model_time_multiclass_preserves_probability_rows():
    from oneehr.config.schema import TaskConfig, TrainerConfig
    from oneehr.data.splits import Split
    from oneehr.models.recurrent import RecurrentTimeModel
    from oneehr.training.trainer import fit_model

    binned = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "bin_time": pd.to_datetime(["2020-01-01", "2020-01-02"] * 4),
            "num__x": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        }
    )
    labels = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "bin_time": pd.to_datetime(["2020-01-01", "2020-01-02"] * 4),
            "label": [0, 1, 1, 2, 2, 0, 0, 2],
            "mask": [1] * 8,
        }
    )
    split = Split(train=np.array(["p1", "p2"], dtype=str), val=np.array(["p3", "p4"], dtype=str), test=np.array([], dtype=str))
    model = RecurrentTimeModel(input_dim=1, hidden_dim=4, out_dim=3, cell="gru")

    _, metrics = fit_model(
        model=model,
        binned=binned,
        split=split,
        feat_cols=["num__x"],
        cfg=TrainerConfig(device="cpu", max_epochs=1, batch_size=2, monitor="val_loss", early_stopping=False),
        task=TaskConfig(kind="multiclass", prediction_mode="time", num_classes=3),
        mode="time",
        labels_df=labels,
    )

    assert "accuracy" in metrics
    assert metrics["history"][0]["val_loss"] >= 0
