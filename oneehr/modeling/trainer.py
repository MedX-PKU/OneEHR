from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch

from oneehr.config.schema import TaskConfig, TrainerConfig
from oneehr.eval.calibration import sigmoid
from oneehr.eval.metrics import binary_metrics, regression_metrics
from oneehr.utils.imports import load_callable


def _select_device(cfg: TrainerConfig):
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("trainer.device='cuda' but CUDA is not available")
        return torch.device("cuda")
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unsupported trainer.device={cfg.device!r}")


def _default_loss(task: TaskConfig):
    nn = torch.nn
    if task.kind == "binary":
        return nn.BCEWithLogitsLoss(reduction="none")
    if task.kind == "regression":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"Unsupported task.kind={task.kind!r}")


def _resolve_loss_fn(task: TaskConfig, trainer: TrainerConfig):
    if trainer.loss_fn is None:
        return _default_loss(task)
    fn = load_callable(trainer.loss_fn)
    loss = fn(task=task)
    return loss


@dataclass(frozen=True)
class FitResult:
    state_dict: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    mask: np.ndarray | None
    history: list[dict[str, float]]


# Keep legacy alias for backward compatibility.
FitSeqResult = FitResult


def _metric_value(task: TaskConfig, monitor: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if monitor.endswith("_loss"):
        raise ValueError("monitor='*_loss' should be computed from loss, not predictions")
    if task.kind == "binary":
        mets = binary_metrics(y_true.astype(float), y_score.astype(float)).metrics
        if monitor in ("val_auc", "val_auroc", "auroc"):
            return float(mets["auroc"])
        if monitor in ("val_auprc", "auprc"):
            return float(mets["auprc"])
        raise ValueError(f"Unsupported monitor metric for binary: {monitor!r}")
    mets = regression_metrics(y_true.astype(float), y_score.astype(float)).metrics
    if monitor in ("val_rmse", "rmse"):
        return float(mets["rmse"])
    if monitor in ("val_mae", "mae"):
        return float(mets["mae"])
    raise ValueError(f"Unsupported monitor metric for regression: {monitor!r}")


def _is_better(mode: str, a: float, b: float) -> bool:
    if mode == "min":
        return a < b
    if mode == "max":
        return a > b
    raise ValueError(f"Unsupported monitor_mode={mode!r}")


def fit_model(
    model,
    X_train,
    len_train,
    y_train,
    static_train,
    X_val,
    len_val,
    y_val,
    static_val,
    task: TaskConfig,
    trainer: TrainerConfig,
    *,
    mask_train: torch.Tensor | None = None,
    mask_val: torch.Tensor | None = None,
) -> FitResult:
    """Unified trainer for both N-1 (patient) and N-N (time) sequence models.

    When *mask_train* / *mask_val* are ``None``, this behaves as a patient-level
    trainer (loss = mean over all samples).  When masks are provided, only the
    masked time-steps contribute to the loss (loss = sum(losses * mask) / sum(mask)).
    """
    masked = mask_train is not None

    device = _select_device(trainer)

    torch.manual_seed(trainer.seed)
    np.random.seed(trainer.seed)

    model = model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay
    )
    loss_fn = _resolve_loss_fn(task, trainer)

    best_state = None
    best_score = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    def _run_epoch(train: bool):
        model.train(train)
        X = X_train if train else X_val
        L = len_train if train else len_val
        y = y_train if train else y_val
        m = (mask_train if train else mask_val) if masked else None
        S = static_train if train else static_val

        idx = torch.randperm(X.shape[0]) if train else torch.arange(X.shape[0])
        total_loss = 0.0
        total_denom = 0.0
        for i in range(0, len(idx), trainer.batch_size):
            b = idx[i : i + trainer.batch_size]
            xb, lb, yb = X[b].to(device), L[b].to(device), y[b].to(device)
            sb = None if S is None else S[b].to(device)
            logits = model(xb, lb) if sb is None else model(xb, lb, sb)
            logits = logits.squeeze(-1)
            losses = loss_fn(logits, yb)

            if m is not None:
                mb = m[b].to(device)
                losses = losses * mb
                denom = mb.sum().clamp_min(1.0)
                loss = losses.sum() / denom
            else:
                loss = losses.mean()
                denom = torch.tensor(float(len(b)))

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip_norm)
                optim.step()
            total_loss += float(loss.detach().cpu()) * float(denom.detach().cpu())
            total_denom += float(denom.detach().cpu())
        return total_loss / max(total_denom, 1.0)

    def _flat_valid(y_np: np.ndarray, yhat_np: np.ndarray, m_np: np.ndarray):
        flat = m_np.reshape(-1).astype(bool)
        return y_np.reshape(-1)[flat], yhat_np.reshape(-1)[flat]

    for _epoch in range(trainer.max_epochs):
        train_loss = _run_epoch(train=True)
        val_loss = _run_epoch(train=False)

        model.eval()
        with torch.no_grad():
            Xv = X_val.to(device)
            Lv = len_val.to(device)
            Sv = None if static_val is None else static_val.to(device)
            val_logits = model(Xv, Lv) if Sv is None else model(Xv, Lv, Sv)
            val_logits = val_logits.squeeze(-1)
            val_logits_np = val_logits.detach().cpu().numpy()

        if task.kind == "binary":
            val_score = sigmoid(val_logits_np)
        else:
            val_score = val_logits_np

        y_val_np = y_val.detach().cpu().numpy()
        if masked:
            m_val_np = mask_val.detach().cpu().numpy()
            yv, sv = _flat_valid(y_val_np, val_score, m_val_np)
        else:
            yv, sv = y_val_np, val_score

        row: dict[str, float] = {"train_loss": float(train_loss), "val_loss": float(val_loss)}
        if trainer.monitor != "val_loss":
            row[trainer.monitor] = _metric_value(task, trainer.monitor, yv, sv)
        history.append(row)

        if trainer.monitor == "val_loss":
            score = float(val_loss)
        else:
            score = float(row[trainer.monitor])

        if best_score is None or _is_better(trainer.monitor_mode, score, best_score):
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if trainer.early_stopping and bad_epochs >= trainer.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        Xv = X_val.to(device)
        Lv = len_val.to(device)
        Sv = None if static_val is None else static_val.to(device)
        logits = model(Xv, Lv) if Sv is None else model(Xv, Lv, Sv)
        logits = logits.squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        y_pred = sigmoid(logits)
    else:
        y_pred = logits

    return FitResult(
        state_dict=model.state_dict(),
        y_true=y_val.detach().cpu().numpy(),
        y_pred=y_pred,
        mask=mask_val.detach().cpu().numpy() if mask_val is not None else None,
        history=history,
    )


def fit_sequence_model(
    model,
    X_train,
    len_train,
    y_train,
    static_train,
    X_val,
    len_val,
    y_val,
    static_val,
    task: TaskConfig,
    trainer: TrainerConfig,
) -> FitResult:
    """Backward-compatible wrapper — delegates to :func:`fit_model`."""
    return fit_model(
        model, X_train, len_train, y_train, static_train,
        X_val, len_val, y_val, static_val, task, trainer,
    )


def fit_sequence_model_time(
    model,
    X_train,
    len_train,
    y_train,
    mask_train,
    static_train,
    X_val,
    len_val,
    y_val,
    mask_val,
    static_val,
    task: TaskConfig,
    trainer: TrainerConfig,
) -> FitResult:
    """Backward-compatible wrapper — delegates to :func:`fit_model`."""
    return fit_model(
        model, X_train, len_train, y_train, static_train,
        X_val, len_val, y_val, static_val, task, trainer,
        mask_train=mask_train, mask_val=mask_val,
    )
