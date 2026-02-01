from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from oneehr.utils.imports import optional_import


def _torch():
    torch = optional_import("torch")
    if torch is None:
        raise ModuleNotFoundError("torch")
    return torch


torch = None
nn = None


def _require_torch():
    global torch, nn
    if torch is not None and nn is not None:
        return
    torch = _torch()
    nn = torch.nn

from oneehr.config.schema import GRUConfig, RNNConfig, TaskConfig, TransformerConfig


class SequenceModel(Protocol):
    def __call__(self, x, lengths): ...


@dataclass(frozen=True)
class FitResult:
    state_dict: dict
    y_true: np.ndarray
    y_pred: np.ndarray


def _make_loss(task: TaskConfig):
    _require_torch()
    if task.kind == "binary":
        return nn.BCEWithLogitsLoss(reduction="none")
    if task.kind == "regression":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"Unsupported task.kind={task.kind!r}")


def fit_sequence_model(
    model,
    X_train,
    len_train,
    y_train,
    X_val,
    len_val,
    y_val,
    task: TaskConfig,
    cfg: GRUConfig | RNNConfig | TransformerConfig,
    device: str = "cpu",
) -> FitResult:
    """Generic trainer for sequence models with (x, lengths) signature.

    For now we reuse the model config as a minimal training config (lr, batch_size, epochs, patience).
    Later we can introduce a dedicated TrainerConfig.
    """

    _require_torch()
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = _make_loss(task)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    def run_epoch(train: bool):
        model.train(train)
        X = X_train if train else X_val
        L = len_train if train else len_val
        y = y_train if train else y_val
        idx = torch.randperm(X.shape[0]) if train else torch.arange(X.shape[0])
        total = 0.0
        count = 0
        for i in range(0, len(idx), cfg.batch_size):
            b = idx[i : i + cfg.batch_size]
            xb, lb, yb = X[b].to(device), L[b].to(device), y[b].to(device)
            logits = model(xb, lb).squeeze(-1)
            losses = loss_fn(logits, yb)
            loss = losses.mean()
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
            total += float(loss.detach().cpu()) * len(b)
            count += len(b)
        return total / max(count, 1)

    for _epoch in range(cfg.max_epochs):
        _ = run_epoch(train=True)
        val_loss = run_epoch(train=False)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device), len_val.to(device)).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        y_pred = 1.0 / (1.0 + np.exp(-logits))
    else:
        y_pred = logits

    return FitResult(state_dict=model.state_dict(), y_true=y_val.detach().cpu().numpy(), y_pred=y_pred)
