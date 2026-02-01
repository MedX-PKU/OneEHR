from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from oneehr.config.schema import GRUConfig, TaskConfig
from oneehr.models.gru import GRUArtifacts, GRUModel


@dataclass(frozen=True)
class GRUPreds:
    y_true: np.ndarray
    y_pred: np.ndarray


def _make_loss(task: TaskConfig):
    if task.kind == "binary":
        return nn.BCEWithLogitsLoss(reduction="none")
    if task.kind == "regression":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"Unsupported task.kind={task.kind!r}")


def train_gru_patient(
    feature_columns: list[str],
    X_train: torch.Tensor,
    len_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    len_val: torch.Tensor,
    y_val: torch.Tensor,
    task: TaskConfig,
    cfg: GRUConfig,
    device: str = "cpu",
) -> tuple[GRUArtifacts, GRUPreds]:
    model = GRUModel(
        input_dim=X_train.shape[-1],
        hidden_dim=cfg.hidden_dim,
        out_dim=1,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

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
    art = GRUArtifacts(feature_columns=feature_columns, state_dict=model.state_dict())
    preds = GRUPreds(y_true=y_val.detach().cpu().numpy(), y_pred=y_pred)
    return art, preds

