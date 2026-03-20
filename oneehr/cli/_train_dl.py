"""Deep-learning model training for a single split (patient + time level)."""
from __future__ import annotations

import numpy as np
import torch

from oneehr.eval.calibration import sigmoid
from oneehr.data.tabular import dynamic_feature_columns, has_static_branch


def _align_static(pids, static, model_supports_static_branch, X_seq):
    """Prepare static features for DL training.

    Returns ``(X_seq_out, S_train_all)`` where *S_train_all* is a tensor when
    the model has a dedicated static branch, and ``None`` otherwise (in which
    case static features have been concatenated into *X_seq_out*).
    """
    from oneehr.data.sequence import align_static_features

    if static is None:
        return X_seq, None

    S_all = align_static_features(
        pids, static, expected_feature_columns=list(static.columns),
    )
    if S_all is None:
        return X_seq, None

    if model_supports_static_branch:
        return X_seq, torch.from_numpy(np.asarray(S_all, dtype=np.float32).copy())

    # Concatenate static into dynamic tensor.
    S_np = np.asarray(S_all, dtype=np.float32)
    S_rep = np.repeat(S_np[:, None, :], X_seq.shape[1], axis=1)
    X_aug = torch.from_numpy(
        np.concatenate([X_seq, S_rep], axis=-1).astype(np.float32, copy=False)
    )
    return X_aug, None


def train_dl_patient_level(
    model,
    binned,
    y,
    static,
    split,
    cfg,
    task,
    model_supports_static_branch: bool = False,
    *,
    y_map: dict[str, float] | None = None,
):
    """Train a DL model in patient-level (N-1) mode for one split.

    Returns ``(y_score, y_true, test_pids, test_logits, val_score, val_logits, y_val_true)``.
    """
    from oneehr.data.sequence import build_patient_sequences, pad_sequences
    from oneehr.modeling.trainer import fit_model

    feat_cols = dynamic_feature_columns(binned)
    pids, seqs, lens = build_patient_sequences(binned, feat_cols)
    X_seq = pad_sequences(seqs, lens)
    lens_t = torch.from_numpy(lens)

    if y_map is None:
        y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
    y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)
    pids_arr = np.array(pids, dtype=str)

    train_m = np.isin(pids_arr, split.train_patients)
    val_m = np.isin(pids_arr, split.val_patients)
    test_m = np.isin(pids_arr, split.test_patients)

    finite_y = np.isfinite(y_arr)
    train_m = train_m & finite_y
    val_m = val_m & finite_y
    test_m = test_m & finite_y

    if not bool(train_m.any()) or not bool(val_m.any()) or not bool(test_m.any()):
        raise SystemExit(
            "No samples available for DL sequence training in this split. "
            "Check split configuration (train/val/test sizes)."
        )

    # Static alignment.
    X_seq, S = _align_static(pids, static, model_supports_static_branch, X_seq)

    X_tr, L_tr, y_tr = X_seq[train_m], lens_t[train_m], torch.from_numpy(y_arr[train_m])
    X_va, L_va, y_va = X_seq[val_m], lens_t[val_m], torch.from_numpy(y_arr[val_m])
    X_te, L_te, y_te = X_seq[test_m], lens_t[test_m], torch.from_numpy(y_arr[test_m])

    if S is not None:
        S_tr, S_va, S_te = S[train_m], S[val_m], S[test_m]
    else:
        S_tr = S_va = S_te = None

    fit = fit_model(
        model=model,
        X_train=X_tr, len_train=L_tr, y_train=y_tr, static_train=S_tr,
        X_val=X_va, len_val=L_va, y_val=y_va, static_val=S_va,
        task=task, trainer=cfg,
    )

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_va is None:
            val_logits = model(X_va, L_va).squeeze(-1).detach().cpu().numpy()
        else:
            val_logits = model(X_va, L_va, S_va).squeeze(-1).detach().cpu().numpy()
    val_score = sigmoid(val_logits) if task.kind == "binary" else val_logits

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_te is None:
            logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
        else:
            logits = model(X_te, L_te, S_te).squeeze(-1).detach().cpu().numpy()
    y_true = y_te.detach().cpu().numpy()
    if task.kind == "binary":
        y_score = sigmoid(logits)
        return y_score, y_true, pids_arr[test_m], logits, val_score, val_logits, y_va.detach().cpu().numpy()
    y_score = logits
    return y_score, y_true, pids_arr[test_m], None, val_score, val_logits, y_va.detach().cpu().numpy()


def train_dl_time_level(
    model,
    binned,
    labels_df,
    static,
    split,
    cfg,
    task,
    model_supports_static_branch: bool = False,
):
    """Train a DL model in time-level (N-N) mode for one split.

    Returns ``(y_score, y_true, key_rows, test_logits, val_score, val_logits, y_val_true)``.
    """
    from oneehr.data.sequence import build_time_sequences, pad_sequences
    from oneehr.modeling.trainer import fit_model

    feat_cols = dynamic_feature_columns(binned)
    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
        binned, labels_df, feat_cols, label_time_col="bin_time",
    )
    X_seq = pad_sequences(seqs, lens)
    Y_seq = pad_sequences([y[:, None] for y in y_seqs], lens).squeeze(-1)
    M_seq = pad_sequences([m[:, None] for m in mask_seqs], lens).squeeze(-1)
    lens_t = torch.from_numpy(lens)

    pids_arr = np.array(pids, dtype=str)
    train_m = np.isin(pids_arr, split.train_patients)
    val_m = np.isin(pids_arr, split.val_patients)
    test_m = np.isin(pids_arr, split.test_patients)

    # Static alignment.
    X_seq, S = _align_static(pids, static, model_supports_static_branch, X_seq)

    X_tr, L_tr, Y_tr, M_tr = X_seq[train_m], lens_t[train_m], Y_seq[train_m], M_seq[train_m]
    X_va, L_va, Y_va, M_va = X_seq[val_m], lens_t[val_m], Y_seq[val_m], M_seq[val_m]
    X_te, L_te, Y_te, M_te = X_seq[test_m], lens_t[test_m], Y_seq[test_m], M_seq[test_m]

    if S is not None:
        S_tr, S_va, S_te = S[train_m], S[val_m], S[test_m]
    else:
        S_tr = S_va = S_te = None

    fit = fit_model(
        model=model,
        X_train=X_tr, len_train=L_tr, y_train=Y_tr, static_train=S_tr,
        X_val=X_va, len_val=L_va, y_val=Y_va, static_val=S_va,
        task=task, trainer=cfg,
        mask_train=M_tr, mask_val=M_va,
    )

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_va is None:
            val_logits = model(X_va, L_va).squeeze(-1).detach().cpu().numpy()
        else:
            val_logits = model(X_va, L_va, S_va).squeeze(-1).detach().cpu().numpy()
    val_score = sigmoid(val_logits) if task.kind == "binary" else val_logits
    y_val_np = Y_va.detach().cpu().numpy()
    m_val_np = M_va.detach().cpu().numpy() if M_va is not None else None
    if m_val_np is not None:
        flat = m_val_np.reshape(-1).astype(bool)
        val_score = val_score.reshape(-1)[flat]
        val_logits = val_logits.reshape(-1)[flat]
        y_val_np = y_val_np.reshape(-1)[flat]

    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        if S_te is None:
            logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
        else:
            logits = model(X_te, L_te, S_te).squeeze(-1).detach().cpu().numpy()
    y_score = sigmoid(logits) if task.kind == "binary" else logits
    y_true = Y_te.detach().cpu().numpy()
    mask = M_te.detach().cpu().numpy()
    if mask is not None:
        flat = mask.reshape(-1).astype(bool)
        y_score = y_score.reshape(-1)[flat]
        y_true = y_true.reshape(-1)[flat]
        logits = logits.reshape(-1)[flat]

    key_rows = []
    for pid, t, m in zip(pids, time_seqs, mask_seqs, strict=True):
        for tt, mm in zip(t, m, strict=True):
            if bool(mm):
                key_rows.append((str(pid), tt))

    if task.kind == "binary":
        return y_score, y_true, key_rows, logits, val_score, val_logits, y_val_np
    return y_score, y_true, key_rows, None, val_score, val_logits, y_val_np
