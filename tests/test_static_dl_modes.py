import numpy as np
import pandas as pd
import torch

from oneehr.cli.train import _train_sequence_patient_level


class _SpyModel(torch.nn.Module):
    def __init__(self, input_dim: int, static_dim: int = 0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.static_dim = int(static_dim)
        self.seen_x_dim: int | None = None
        self.seen_static_dim: int | None = None
        self._dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        self.seen_x_dim = int(x.shape[-1])
        if static is None:
            self.seen_static_dim = None
        else:
            self.seen_static_dim = int(static.shape[-1])
        # return logits shape (B, 1)
        return self._dummy * 0.0 + torch.zeros((x.shape[0], 1), dtype=torch.float32)


def _tiny_binned() -> pd.DataFrame:
    rows = []
    for pid in ["p1", "p2", "p3"]:
        for t in [0, 1]:
            rows.append(
                {
                    "patient_id": pid,
                    "bin_time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=t),
                    "label": 0.0,
                    "num__A": float(t),
                    "cat__B__x": 1.0,
                }
            )
    return pd.DataFrame(rows)


class _Split:
    train_patients = ["p1"]
    val_patients = ["p2"]
    test_patients = ["p3"]


def test_static_branch_passes_static_tensor():
    binned = _tiny_binned()
    y = pd.Series([0.0, 1.0, 0.0], index=["p1", "p2", "p3"])
    static = pd.DataFrame({"s1": [1.0, 2.0, 3.0]}, index=["p1", "p2", "p3"])
    model = _SpyModel(input_dim=2, static_dim=1)

    # Use a tiny trainer config by reusing defaults (device=cpu) is handled inside trainer.
    from oneehr.config.schema import TrainerConfig, TaskConfig

    task = TaskConfig(kind="binary", prediction_mode="patient")
    trainer = TrainerConfig(device="cpu", max_epochs=1, batch_size=8, early_stopping=False)

    _train_sequence_patient_level(
        model=model,
        binned=binned,
        y=y,
        static=static,
        split=_Split(),
        cfg=trainer,
        task=task,
        model_supports_static_branch=True,
    )

    assert model.seen_x_dim == 2
    assert model.seen_static_dim == 1


def test_static_concat_increases_dynamic_dim():
    binned = _tiny_binned()
    y = pd.Series([0.0, 1.0, 0.0], index=["p1", "p2", "p3"])
    static = pd.DataFrame({"s1": [1.0, 2.0, 3.0]}, index=["p1", "p2", "p3"])
    model = _SpyModel(input_dim=3, static_dim=0)

    from oneehr.config.schema import TrainerConfig, TaskConfig

    task = TaskConfig(kind="binary", prediction_mode="patient")
    trainer = TrainerConfig(device="cpu", max_epochs=1, batch_size=8, early_stopping=False)

    _train_sequence_patient_level(
        model=model,
        binned=binned,
        y=y,
        static=static,
        split=_Split(),
        cfg=trainer,
        task=task,
        model_supports_static_branch=False,
    )

    assert model.seen_x_dim == 3
    assert model.seen_static_dim is None
