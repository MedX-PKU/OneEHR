"""Unified DL model trainer."""
from __future__ import annotations

import numpy as np
import torch

from oneehr.config.schema import TaskConfig, TrainerConfig


def _select_device(cfg: TrainerConfig):
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("trainer.device='cuda' but CUDA is not available")
        return torch.device("cuda")
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported trainer.device={cfg.device!r}")


def _default_loss(task: TaskConfig):
    if task.kind == "binary":
        return torch.nn.BCEWithLogitsLoss(reduction="none")
    if task.kind == "regression":
        return torch.nn.MSELoss(reduction="none")
    raise ValueError(f"Unsupported task.kind={task.kind!r}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _metric_value(task: TaskConfig, monitor: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the monitored metric from val predictions."""
    from oneehr.eval.metrics import binary_metrics, regression_metrics

    if task.kind == "binary":
        mets = binary_metrics(y_true.astype(float), y_score.astype(float)).metrics
        if monitor in ("val_auroc", "auroc"):
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
    """Compare two scores based on monitor_mode."""
    if mode == "min":
        return a < b
    if mode == "max":
        return a > b
    raise ValueError(f"Unsupported monitor_mode={mode!r}")


def fit_model(
    *,
    model,
    binned,
    split,
    feat_cols: list[str],
    cfg: TrainerConfig,
    task: TaskConfig,
    mode: str = "patient",
    y_map: dict | None = None,
    labels_df=None,
    static=None,
    train_extra: dict[str, torch.Tensor] | None = None,
    val_extra: dict[str, torch.Tensor] | None = None,
) -> tuple[object, dict]:
    """Train a DL model and return (trained_model, val_metrics_dict).

    Handles both patient-level and time-level training.
    """
    from oneehr.data.sequence import build_patient_sequences, build_time_sequences, pad_sequences
    from oneehr.data.tabular import has_static_branch
    from oneehr.eval.metrics import binary_metrics, regression_metrics

    from oneehr.utils import set_seed

    device = _select_device(cfg)
    set_seed(cfg.seed)

    model_supports_static = has_static_branch(model)
    use_static = static is not None and model_supports_static

    # Filter binned to train/val patients
    binned_train = binned[binned["patient_id"].astype(str).isin(set(split.train))].copy()
    binned_val = binned[binned["patient_id"].astype(str).isin(set(split.val))].copy()

    if mode == "patient":
        X_train, len_train, y_train, static_train = _prep_patient(
            binned_train, feat_cols, y_map or {}, split.train, static if use_static else None,
        )
        X_val, len_val, y_val, static_val = _prep_patient(
            binned_val, feat_cols, y_map or {}, split.val, static if use_static else None,
        )
        mask_train = mask_val = None
    else:
        if labels_df is None:
            raise ValueError("labels_df required for time-level training")
        labels_train = labels_df[labels_df["patient_id"].astype(str).isin(set(split.train))]
        labels_val = labels_df[labels_df["patient_id"].astype(str).isin(set(split.val))]

        X_train, len_train, y_train, mask_train, static_train = _prep_time(
            binned_train, labels_train, feat_cols, split.train, static if use_static else None,
        )
        X_val, len_val, y_val, mask_val, static_val = _prep_time(
            binned_val, labels_val, feat_cols, split.val, static if use_static else None,
        )

    # Training loop
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = _default_loss(task)

    best_state = None
    best_score: float | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    def _val_predictions():
        """Compute val predictions for metric evaluation."""
        model.eval()
        with torch.no_grad():
            Xv = X_val.to(device)
            Lv = len_val.to(device)
            Sv = None if static_val is None else static_val.to(device)
            kw = {}
            if val_extra:
                kw = {
                    k: v.to(device) if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                    for k, v in val_extra.items()
                }
            logits = model(Xv, Lv, **kw) if Sv is None else model(Xv, Lv, Sv, **kw)
            logits = logits.squeeze(-1).detach().cpu().numpy()

        if task.kind == "binary":
            y_score = sigmoid(logits)
        else:
            y_score = logits

        y_val_np = y_val.detach().cpu().numpy()
        if mask_val is not None:
            m_np = mask_val.detach().cpu().numpy()
            if y_score.ndim > 1 and y_val_np.ndim > 1 and y_score.shape[-1] != y_val_np.shape[-1]:
                t = y_score.shape[-1]
                y_val_np = y_val_np[:, :t]
                m_np = m_np[:, :t]
            m = m_np.reshape(-1).astype(bool)
            return y_val_np.reshape(-1)[m], y_score.reshape(-1)[m]
        return y_val_np, y_score

    monitor_needs_preds = cfg.monitor != "val_loss"

    for epoch in range(cfg.max_epochs):
        # Train
        model.train()
        train_loss = _run_epoch(
            model, X_train, len_train, y_train, static_train, mask_train,
            loss_fn, optim, cfg, device, train=True, extra=train_extra,
        )
        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                model, X_val, len_val, y_val, static_val, mask_val,
                loss_fn, None, cfg, device, train=False, extra=val_extra,
            )

        row: dict[str, float] = {"train_loss": float(train_loss), "val_loss": float(val_loss)}

        if monitor_needs_preds:
            yv, sv = _val_predictions()
            row[cfg.monitor] = _metric_value(task, cfg.monitor, yv, sv)

        history.append(row)

        if cfg.monitor == "val_loss":
            score = float(val_loss)
        else:
            score = float(row[cfg.monitor])

        if best_score is None or _is_better(cfg.monitor_mode, score, best_score):
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if cfg.early_stopping and bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute final val metrics from best model
    yv_final, yp_final = _val_predictions()

    if task.kind == "binary":
        metrics = binary_metrics(yv_final.astype(float), yp_final.astype(float)).metrics
    else:
        metrics = regression_metrics(yv_final.astype(float), yp_final.astype(float)).metrics

    metrics["history"] = history

    model = model.cpu()
    return model, metrics


def _run_epoch(model, X, L, y, S, mask, loss_fn, optim, cfg, device, *, train: bool, extra=None) -> float:
    idx = torch.randperm(X.shape[0]) if train else torch.arange(X.shape[0])
    total_loss = 0.0
    total_denom = 0.0
    bs = cfg.batch_size

    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        xb = X[b].to(device)
        lb = L[b].to(device)
        yb = y[b].to(device)
        sb = None if S is None else S[b].to(device)

        kw = {}
        if extra:
            kw = {
                k: v[b].to(device) if isinstance(v, torch.Tensor) and v.dim() > 1 else (
                    v.to(device) if isinstance(v, torch.Tensor) else v
                )
                for k, v in extra.items()
            }
        logits = model(xb, lb, **kw) if sb is None else model(xb, lb, sb, **kw)
        logits = logits.squeeze(-1)
        # Align time dimensions: pack/unpack may return shorter sequences
        mb = mask[b].to(device) if mask is not None else None
        if logits.ndim > 1 and yb.ndim > 1 and logits.shape[-1] != yb.shape[-1]:
            t = logits.shape[-1]
            yb = yb[:, :t]
            if mb is not None:
                mb = mb[:, :t]
        losses = loss_fn(logits, yb)

        if mb is not None:
            losses = losses * mb
            denom = mb.sum().clamp_min(1.0)
            loss = losses.sum() / denom
        else:
            loss = losses.mean()
            denom = torch.tensor(float(len(b)))

        if train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

        total_loss += float(loss.detach().cpu()) * float(denom.detach().cpu())
        total_denom += float(denom.detach().cpu())

    return total_loss / max(total_denom, 1.0)


def _prep_patient(binned, feat_cols, y_map, patient_ids, static):
    from oneehr.data.sequence import build_patient_sequences, pad_sequences

    pids, seqs, lens = build_patient_sequences(binned, feat_cols)
    X_seq = pad_sequences(seqs, lens)
    lens_t = torch.from_numpy(lens)

    # Build y from y_map
    y = torch.tensor([float(y_map.get(str(p), float("nan"))) for p in pids], dtype=torch.float32)

    # Filter out NaN labels
    valid = ~torch.isnan(y)
    X_seq = X_seq[valid]
    lens_t = lens_t[valid]
    y = y[valid]
    valid_pids = [p for p, v in zip(pids, valid.tolist()) if v]

    static_t = None
    if static is not None:
        import numpy as np
        static_arr = static.reindex(index=np.array(valid_pids, dtype=str)).to_numpy(dtype=np.float32, copy=True)
        static_t = torch.from_numpy(static_arr)

    return X_seq, lens_t, y, static_t


def _prep_time(binned, labels_df, feat_cols, patient_ids, static):
    from oneehr.data.sequence import build_time_sequences, pad_sequences
    import numpy as np

    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
        binned, labels_df, feat_cols, label_time_col="bin_time",
    )
    X_seq = pad_sequences(seqs, lens)
    Y_seq = pad_sequences([yy[:, None] for yy in y_seqs], lens).squeeze(-1)
    M_seq = pad_sequences([mm[:, None] for mm in mask_seqs], lens).squeeze(-1)
    lens_t = torch.from_numpy(lens)

    static_t = None
    if static is not None:
        static_arr = static.reindex(index=np.array(pids, dtype=str)).to_numpy(dtype=np.float32, copy=True)
        static_t = torch.from_numpy(static_arr)

    return X_seq, lens_t, Y_seq, M_seq, static_t
