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


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class FocalLoss(torch.nn.Module):
    """Binary focal loss (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal = ((1 - pt) ** self.gamma) * bce
        if self.weight is not None:
            w = targets * self.weight[1] + (1 - targets) * self.weight[0]
            focal = focal * w
        return focal


class FocalLossMulticlass(torch.nn.Module):
    """Multiclass focal loss."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


def _build_loss(task: TaskConfig, class_weights: torch.Tensor | None = None):
    if task.kind == "binary":
        if task.loss == "focal":
            return FocalLoss(gamma=task.focal_gamma, weight=class_weights)
        return torch.nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=class_weights[1] / class_weights[0] if class_weights is not None else None,
        )
    if task.kind == "multiclass":
        if task.loss == "focal":
            return FocalLossMulticlass(gamma=task.focal_gamma, weight=class_weights)
        return torch.nn.CrossEntropyLoss(reduction="none", weight=class_weights)
    if task.kind == "multilabel":
        return torch.nn.BCEWithLogitsLoss(reduction="none")
    if task.kind == "regression":
        return torch.nn.MSELoss(reduction="none")
    raise ValueError(f"Unsupported task.kind={task.kind!r}")


def _compute_class_weights(y: torch.Tensor, task: TaskConfig) -> torch.Tensor | None:
    """Compute balanced class weights from label distribution."""
    y_np = y.detach().cpu().numpy().reshape(-1)
    y_np = y_np[np.isfinite(y_np)]
    if task.kind == "binary":
        n_pos = (y_np == 1).sum()
        n_neg = (y_np == 0).sum()
        if n_pos == 0 or n_neg == 0:
            return None
        total = n_pos + n_neg
        return torch.tensor([total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float32)
    if task.kind == "multiclass":
        classes = np.unique(y_np.astype(int))
        n_classes = int(classes.max()) + 1
        counts = np.bincount(y_np.astype(int), minlength=n_classes).astype(float)
        counts = np.maximum(counts, 1.0)
        weights = len(y_np) / (n_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)
    return None


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


def _build_scheduler(cfg: TrainerConfig, optimizer: torch.optim.Optimizer):
    if cfg.scheduler == "none":
        return None
    params = cfg.scheduler_params
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get("T_max", cfg.max_epochs),
            eta_min=params.get("eta_min", 0),
        )
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(params.get("step_size", 10)),
            gamma=float(params.get("gamma", 0.1)),
        )
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.monitor_mode,
            factor=float(params.get("factor", 0.1)),
            patience=int(params.get("patience", 5)),
        )
    raise ValueError(f"Unsupported trainer.scheduler={cfg.scheduler!r}")


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


def _get_amp_dtype(cfg: TrainerConfig):
    if cfg.precision == "fp16":
        return torch.float16
    if cfg.precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    return None  # fp32 — no autocast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for 2-D arrays."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _metric_value(task: TaskConfig, monitor: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the monitored metric from val predictions."""
    from oneehr.eval.metrics import binary_metrics, multiclass_metrics, regression_metrics

    if task.kind == "binary":
        mets = binary_metrics(y_true.astype(float), y_score.astype(float)).metrics
        if monitor in ("val_auroc", "auroc"):
            return float(mets["auroc"])
        if monitor in ("val_auprc", "auprc"):
            return float(mets["auprc"])
        raise ValueError(f"Unsupported monitor metric for binary: {monitor!r}")
    if task.kind == "multiclass":
        mets = multiclass_metrics(
            y_true.astype(int),
            y_score,
            num_classes=task.num_classes or int(y_true.max()) + 1,
        ).metrics
        if monitor in ("val_auroc", "auroc"):
            return float(mets["auroc_macro"])
        if monitor in ("val_f1", "f1"):
            return float(mets["f1_macro"])
        raise ValueError(f"Unsupported monitor metric for multiclass: {monitor!r}")
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
    max_seq_length: int | None = None,
) -> tuple[object, dict]:
    """Train a DL model and return (trained_model, val_metrics_dict).

    Handles both patient-level and time-level training.
    """
    from oneehr.data.tabular import has_static_branch
    from oneehr.eval.metrics import binary_metrics, multiclass_metrics, regression_metrics
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
            binned_train,
            feat_cols,
            y_map or {},
            split.train,
            static if use_static else None,
            max_seq_length=max_seq_length,
        )
        X_val, len_val, y_val, static_val = _prep_patient(
            binned_val,
            feat_cols,
            y_map or {},
            split.val,
            static if use_static else None,
            max_seq_length=max_seq_length,
        )
        mask_train = mask_val = None
    else:
        if labels_df is None:
            raise ValueError("labels_df required for time-level training")
        labels_train = labels_df[labels_df["patient_id"].astype(str).isin(set(split.train))]
        labels_val = labels_df[labels_df["patient_id"].astype(str).isin(set(split.val))]

        X_train, len_train, y_train, mask_train, static_train = _prep_time(
            binned_train,
            labels_train,
            feat_cols,
            split.train,
            static if use_static else None,
            max_seq_length=max_seq_length,
        )
        X_val, len_val, y_val, mask_val, static_val = _prep_time(
            binned_val,
            labels_val,
            feat_cols,
            split.val,
            static if use_static else None,
            max_seq_length=max_seq_length,
        )

    # Class weights
    class_weights = None
    if cfg.class_weight == "balanced" and task.kind in ("binary", "multiclass"):
        class_weights = _compute_class_weights(y_train, task)
        if class_weights is not None:
            class_weights = class_weights.to(device)

    # Training loop
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = _build_loss(task, class_weights)
    scheduler = _build_scheduler(cfg, optim)

    # Mixed precision
    amp_dtype = _get_amp_dtype(cfg)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_state = None
    best_score: float | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    def _val_predictions():
        """Compute val predictions for metric evaluation."""
        model.eval()
        with torch.no_grad():
            ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else _nullcontext()
            with ctx:
                Xv = X_val.to(device)
                Lv = len_val.to(device)
                Sv = None if static_val is None else static_val.to(device)
                kw = {}
                if val_extra:
                    kw = {k: v.to(device) if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in val_extra.items()}
                logits = model(Xv, Lv, **kw) if Sv is None else model(Xv, Lv, Sv, **kw)
                logits = logits.squeeze(-1).detach().cpu().float().numpy()

        if task.kind == "binary":
            y_score = sigmoid(logits)
        elif task.kind == "multiclass":
            y_score = softmax(logits)
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

    # Progress bar
    try:
        from tqdm import tqdm

        epoch_iter = tqdm(range(cfg.max_epochs), desc="Training", unit="epoch")
    except ImportError:
        epoch_iter = range(cfg.max_epochs)

    for epoch in epoch_iter:
        # Train
        model.train()
        train_loss = _run_epoch(
            model,
            X_train,
            len_train,
            y_train,
            static_train,
            mask_train,
            loss_fn,
            optim,
            cfg,
            device,
            task,
            train=True,
            extra=train_extra,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                model,
                X_val,
                len_val,
                y_val,
                static_val,
                mask_val,
                loss_fn,
                None,
                cfg,
                device,
                task,
                train=False,
                extra=val_extra,
                amp_dtype=amp_dtype,
                scaler=None,
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

        # Update progress bar
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f"{val_loss:.4f}", best=f"{best_score:.4f}" if best_score is not None else "?")

        if best_score is None or _is_better(cfg.monitor_mode, score, best_score):
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if cfg.early_stopping and bad_epochs >= cfg.patience:
                break

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute final val metrics from best model
    yv_final, yp_final = _val_predictions()

    if task.kind == "binary":
        metrics = binary_metrics(yv_final.astype(float), yp_final.astype(float)).metrics
    elif task.kind == "multiclass":
        metrics = multiclass_metrics(
            yv_final.astype(int),
            yp_final,
            num_classes=task.num_classes or int(yv_final.max()) + 1,
        ).metrics
    else:
        metrics = regression_metrics(yv_final.astype(float), yp_final.astype(float)).metrics

    metrics["history"] = history

    model = model.cpu()
    return model, metrics


def _run_epoch(
    model,
    X,
    L,
    y,
    S,
    mask,
    loss_fn,
    optim,
    cfg,
    device,
    task,
    *,
    train: bool,
    extra=None,
    amp_dtype=None,
    scaler=None,
) -> float:
    idx = torch.randperm(X.shape[0]) if train else torch.arange(X.shape[0])
    total_loss = 0.0
    total_denom = 0.0
    bs = cfg.batch_size
    use_amp = amp_dtype is not None and device.type == "cuda"

    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        xb = X[b].to(device)
        lb = L[b].to(device)
        yb = y[b].to(device)
        sb = None if S is None else S[b].to(device)

        kw = {}
        if extra:
            kw = {k: v[b].to(device) if isinstance(v, torch.Tensor) and v.dim() > 1 else (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in extra.items()}

        ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else _nullcontext()
        with ctx:
            logits = model(xb, lb, **kw) if sb is None else model(xb, lb, sb, **kw)
            logits = logits.squeeze(-1)
            # Align time dimensions: pack/unpack may return shorter sequences
            mb = mask[b].to(device) if mask is not None else None
            if logits.ndim > 1 and yb.ndim > 1 and logits.shape[-1] != yb.shape[-1]:
                t = logits.shape[-1]
                yb = yb[:, :t]
                if mb is not None:
                    mb = mb[:, :t]

            # For multiclass, targets must be long
            if task.kind == "multiclass":
                yb = yb.long()

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
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

        total_loss += float(loss.detach().cpu()) * float(denom.detach().cpu())
        total_denom += float(denom.detach().cpu())

    return total_loss / max(total_denom, 1.0)


class _nullcontext:
    """Minimal no-op context manager (avoid importing contextlib)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _prep_patient(binned, feat_cols, y_map, patient_ids, static, *, max_seq_length=None):
    from oneehr.data.sequence import build_patient_sequences, pad_sequences

    pids, seqs, lens = build_patient_sequences(binned, feat_cols, max_seq_length=max_seq_length)
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


def _prep_time(binned, labels_df, feat_cols, patient_ids, static, *, max_seq_length=None):
    import numpy as np

    from oneehr.data.sequence import build_time_sequences, pad_sequences

    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
        binned,
        labels_df,
        feat_cols,
        label_time_col="bin_time",
        max_seq_length=max_seq_length,
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
