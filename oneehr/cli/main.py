import argparse

import numpy as np

from oneehr.utils.imports import optional_import

from oneehr.config.load import load_experiment_config
from oneehr.data.patient_index import make_patient_index
from oneehr.data.io import load_event_table
from oneehr.data.tabular import make_patient_tabular, make_time_tabular
from oneehr.models.xgb import predict_xgboost, train_xgboost
from oneehr.data.binning import bin_events
from oneehr.data.labels import normalize_patient_labels, normalize_time_labels, run_label_fn
from oneehr.eval.metrics import binary_metrics, regression_metrics
from oneehr.data.sequence import build_patient_sequences
from oneehr.data.splits import make_splits
from oneehr.utils.io import ensure_dir, write_json
from oneehr.eval.tables import summarize_metrics, to_paper_wide_table
from oneehr.hpo.grid import apply_overrides, iter_grid
from oneehr.hpo.runner import select_best_with_trials
from oneehr.modeling.trainer import fit_sequence_model, fit_sequence_model_time


def _train_sequence_patient_level(
    model,
    binned,
    y,
    split,
    cfg,
    task,
):
    torch = optional_import("torch")
    if torch is None:
        raise SystemExit("Missing optional dependency: torch. Install it first (e.g. `uv add torch`).")

    from oneehr.data.sequence import pad_sequences

    feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    pids, seqs, lens = build_patient_sequences(binned, feat_cols)
    X_seq = pad_sequences(seqs, lens)
    lens_t = torch.from_numpy(lens)

    y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
    y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)
    pids_arr = np.array(pids, dtype=str)

    train_m = np.isin(pids_arr, split.train_patients)
    val_m = np.isin(pids_arr, split.val_patients)
    test_m = np.isin(pids_arr, split.test_patients)

    X_tr, L_tr, y_tr = X_seq[train_m], lens_t[train_m], torch.from_numpy(y_arr[train_m])
    X_va, L_va, y_va = X_seq[val_m], lens_t[val_m], torch.from_numpy(y_arr[val_m])
    X_te, L_te, y_te = X_seq[test_m], lens_t[test_m], torch.from_numpy(y_arr[test_m])

    fit = fit_sequence_model(
        model=model,
        X_train=X_tr,
        len_train=L_tr,
        y_train=y_tr,
        X_val=X_va,
        len_val=L_va,
        y_val=y_va,
        task=task,
        trainer=cfg,
    )

    # final evaluation on test
    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        y_score = 1.0 / (1.0 + np.exp(-logits))
    else:
        y_score = logits
    return y_score, y_te.detach().cpu().numpy(), pids_arr[test_m]


def _train_sequence_time_level(
    model,
    binned,
    labels_df,
    split,
    cfg,
    task,
):
    torch = optional_import("torch")
    if torch is None:
        raise SystemExit("Missing optional dependency: torch. Install it first (e.g. `uv add torch`).")

    from oneehr.data.sequence import build_time_sequences, pad_sequences

    feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
    pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
        binned,
        labels_df,
        feat_cols,
        label_time_col="bin_time",
    )
    X_seq = pad_sequences(seqs, lens)
    Y_seq = pad_sequences([y[:, None] for y in y_seqs], lens).squeeze(-1)
    M_seq = pad_sequences([m[:, None] for m in mask_seqs], lens).squeeze(-1)
    lens_t = torch.from_numpy(lens)

    pids_arr = np.array(pids, dtype=str)
    train_m = np.isin(pids_arr, split.train_patients)
    val_m = np.isin(pids_arr, split.val_patients)
    test_m = np.isin(pids_arr, split.test_patients)

    X_tr, L_tr, Y_tr, M_tr = (
        X_seq[train_m],
        lens_t[train_m],
        Y_seq[train_m],
        M_seq[train_m],
    )
    X_va, L_va, Y_va, M_va = (
        X_seq[val_m],
        lens_t[val_m],
        Y_seq[val_m],
        M_seq[val_m],
    )
    X_te, L_te, Y_te, M_te = (
        X_seq[test_m],
        lens_t[test_m],
        Y_seq[test_m],
        M_seq[test_m],
    )

    fit = fit_sequence_model_time(
        model=model,
        X_train=X_tr,
        len_train=L_tr,
        y_train=Y_tr,
        mask_train=M_tr,
        X_val=X_va,
        len_val=L_va,
        y_val=Y_va,
        mask_val=M_va,
        task=task,
        trainer=cfg,
    )

    # final evaluation on test
    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(X_te, L_te).squeeze(-1).detach().cpu().numpy()
    if task.kind == "binary":
        y_score = 1.0 / (1.0 + np.exp(-logits))
    else:
        y_score = logits
    y_true = Y_te.detach().cpu().numpy()
    mask = M_te.detach().cpu().numpy()
    if mask is not None:
        flat = mask.reshape(-1).astype(bool)
        y_score = y_score.reshape(-1)[flat]
        y_true = y_true.reshape(-1)[flat]

    key_rows = []
    for pid, t, m in zip(pids, time_seqs, mask_seqs, strict=True):
        for tt, mm in zip(t, m, strict=True):
            if bool(mm):
                key_rows.append((str(pid), tt))

    return y_score, y_true, key_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oneehr")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Preprocess raw event table into model-ready artifacts")
    p_pre.add_argument("--config", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True)

    p_bm = sub.add_parser("benchmark", help="Run benchmarking across splits/models")
    p_bm.add_argument("--config", required=True)

    p_hpo = sub.add_parser("hpo", help="Run hyperparameter selection on validation")
    p_hpo.add_argument("--config", required=True)

    return parser


def _run_preprocess(cfg_path: str) -> None:
    cfg = load_experiment_config(cfg_path)
    events = load_event_table(cfg.dataset)
    binned = bin_events(events, cfg.dataset, cfg.preprocess)

    labels_res = run_label_fn(events, cfg)

    out_root = cfg.output.root / cfg.output.run_name
    ensure_dir(out_root)
    (out_root / "binned.parquet").write_bytes(binned.table.to_parquet(index=False))
    (out_root / "code_vocab.txt").write_text("\n".join(binned.code_vocab), encoding="utf-8")

    if labels_res is not None:
        if cfg.task.prediction_mode == "patient":
            labels = normalize_patient_labels(labels_res.df)
        elif cfg.task.prediction_mode == "time":
            labels = normalize_time_labels(labels_res.df, cfg)
        else:
            raise SystemExit(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")
        (out_root / "labels.parquet").write_bytes(labels.to_parquet(index=False))


def _run_train(cfg_path: str) -> None:
    cfg = load_experiment_config(cfg_path)
    events = load_event_table(cfg.dataset)
    patient_index = make_patient_index(events, cfg.dataset.time_col, cfg.dataset.patient_id_col)
    splits = make_splits(patient_index, cfg.split)

    binned = bin_events(events, cfg.dataset, cfg.preprocess).table

    labels_res = run_label_fn(events, cfg)
    labels_df = None
    if labels_res is not None:
        if cfg.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(labels_res.df)
        elif cfg.task.prediction_mode == "time":
            labels_df = normalize_time_labels(labels_res.df, cfg)

    split0 = splits[0]

    if cfg.model.name in {"gru", "rnn", "transformer"}:
        torch = optional_import("torch")
        if torch is None:
            raise SystemExit(
                "Missing optional dependency: torch. Install it first (e.g. `uv add torch`)."
            )

    if cfg.task.prediction_mode == "patient":
        if labels_df is not None:
            lab = labels_df.set_index("patient_id")["label"]
            X, _ = make_patient_tabular(binned)
            y = lab.reindex(X.index.astype(str))
        else:
            X, y = make_patient_tabular(binned)
        train_mask = X.index.astype(str).isin(split0.train_patients)
        val_mask = X.index.astype(str).isin(split0.val_patients)
        test_mask = X.index.astype(str).isin(split0.test_patients)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask].to_numpy()
        X_val, y_val = X.loc[val_mask], y.loc[val_mask].to_numpy()
        X_test, y_test = X.loc[test_mask], y.loc[test_mask].to_numpy()
        test_key = None
        test_patient_ids = X_test.index.astype(str)
    elif cfg.task.prediction_mode == "time":
        if labels_df is not None:
            # Join features with labels on (patient_id, bin_time)
            X0, _, key0 = make_time_tabular(binned)
            feat = key0.copy()
            feat["_row"] = feat.index
            lab = labels_df.merge(feat, on=["patient_id", "bin_time"], how="inner")
            X = X0.loc[lab["_row"].to_numpy()].reset_index(drop=True)
            y = lab["label"].reset_index(drop=True)
            key = lab[["patient_id", "bin_time"]].reset_index(drop=True)
            mask = lab["mask"].to_numpy().astype(bool)
        else:
            X, y, key = make_time_tabular(binned)
            mask = None
        # Split by patient_id, but rows are (patient, bin).
        train_mask = key["patient_id"].astype(str).isin(split0.train_patients)
        val_mask = key["patient_id"].astype(str).isin(split0.val_patients)
        test_mask = key["patient_id"].astype(str).isin(split0.test_patients)

        if mask is not None:
            train_mask = train_mask & mask
            val_mask = val_mask & mask
            test_mask = test_mask & mask

        X_train, y_train = X.loc[train_mask], y.loc[train_mask].to_numpy()
        X_val, y_val = X.loc[val_mask], y.loc[val_mask].to_numpy()
        X_test, y_test = X.loc[test_mask], y.loc[test_mask].to_numpy()
        test_key = key.loc[test_mask].reset_index(drop=True)
        test_patient_ids = test_key["patient_id"].astype(str)
    else:
        raise SystemExit(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")

    if cfg.model.name == "xgboost":
        art = train_xgboost(X_train, y_train, X_val, y_val, cfg.task, cfg.model.xgboost)
        y_score = predict_xgboost(art, X_test, cfg.task)
    else:
        if cfg.task.prediction_mode == "patient":
            pass
        elif cfg.task.prediction_mode == "time":
            if labels_df is None:
                raise SystemExit("prediction_mode='time' requires labels (labels.fn).")
        else:
            raise SystemExit(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")

        input_dim = len(
            [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
        )

        if cfg.model.name == "gru":
            from oneehr.models.gru import GRUModel
            from oneehr.models.gru import GRUTimeModel

            if cfg.task.prediction_mode == "patient":
                model = GRUModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.gru.hidden_dim,
                    out_dim=1,
                    num_layers=cfg.model.gru.num_layers,
                    dropout=cfg.model.gru.dropout,
                )
                y_score, y_test, test_patient_ids = _train_sequence_patient_level(
                    model=model,
                    binned=binned,
                    y=y,
                    split=split0,
                    cfg=cfg.trainer,
                    task=cfg.task,
                )
            else:
                model = GRUTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.gru.hidden_dim,
                    out_dim=1,
                    num_layers=cfg.model.gru.num_layers,
                    dropout=cfg.model.gru.dropout,
                )
                y_score, y_test, test_key_rows = _train_sequence_time_level(
                    model=model,
                    binned=binned,
                    labels_df=labels_df,
                    split=split0,
                    cfg=cfg.trainer,
                    task=cfg.task,
                )
                test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])
        elif cfg.model.name == "rnn":
            from oneehr.models.rnn import RNNModel
            from oneehr.models.rnn import RNNTimeModel

            if cfg.task.prediction_mode == "patient":
                model = RNNModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.rnn.hidden_dim,
                    out_dim=1,
                    num_layers=cfg.model.rnn.num_layers,
                    dropout=cfg.model.rnn.dropout,
                    bidirectional=cfg.model.rnn.bidirectional,
                    nonlinearity=cfg.model.rnn.nonlinearity,
                )
                y_score, y_test, test_patient_ids = _train_sequence_patient_level(
                    model=model,
                    binned=binned,
                    y=y,
                    split=split0,
                    cfg=cfg.trainer,
                    task=cfg.task,
                )
            else:
                model = RNNTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.rnn.hidden_dim,
                    out_dim=1,
                    num_layers=cfg.model.rnn.num_layers,
                    dropout=cfg.model.rnn.dropout,
                    bidirectional=cfg.model.rnn.bidirectional,
                    nonlinearity=cfg.model.rnn.nonlinearity,
                )
                y_score, y_test, test_key_rows = _train_sequence_time_level(
                    model=model,
                    binned=binned,
                    labels_df=labels_df,
                    split=split0,
                    cfg=cfg.trainer,
                    task=cfg.task,
                )
                test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])
        elif cfg.model.name == "transformer":
            from oneehr.models.transformer import TransformerModel
            from oneehr.models.transformer import TransformerTimeModel

            if cfg.task.prediction_mode == "patient":
                model = TransformerModel(
                    input_dim=input_dim,
                    d_model=cfg.model.transformer.d_model,
                    out_dim=1,
                    nhead=cfg.model.transformer.nhead,
                    num_layers=cfg.model.transformer.num_layers,
                    dim_feedforward=cfg.model.transformer.dim_feedforward,
                    dropout=cfg.model.transformer.dropout,
                    pooling=cfg.model.transformer.pooling,
                )
                y_score, y_test, test_patient_ids = _train_sequence_patient_level(
                    model=model,
                    binned=binned,
                    y=y,
                    split=split0,
                    cfg=cfg.trainer,
                    task=cfg.task,
                )
            else:
                model = TransformerTimeModel(
                    input_dim=input_dim,
                    d_model=cfg.model.transformer.d_model,
                    out_dim=1,
                    nhead=cfg.model.transformer.nhead,
                    num_layers=cfg.model.transformer.num_layers,
                    dim_feedforward=cfg.model.transformer.dim_feedforward,
                    dropout=cfg.model.transformer.dropout,
                )
                y_score, y_test, test_key_rows = _train_sequence_time_level(
                    model=model,
                    binned=binned,
                    labels_df=labels_df,
                    split=split0,
                    cfg=cfg.trainer,
                    task=cfg.task,
                )
                test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])
        else:
            raise SystemExit(f"Unsupported model.name={cfg.model.name!r}")

        test_key = None

    if cfg.task.kind == "binary":
        metrics = binary_metrics(y_test.astype(float), y_score.astype(float)).metrics
    else:
        metrics = regression_metrics(y_test.astype(float), y_score.astype(float)).metrics

    out_root = cfg.output.root / cfg.output.run_name
    ensure_dir(out_root)
    write_json(out_root / "metrics.json", metrics)

    if cfg.output.save_preds and len(test_patient_ids) > 0:
        import pandas as pd

        preds = pd.DataFrame({"patient_id": test_patient_ids, "y_true": y_test, "y_pred": y_score})
        if test_key is not None:
            preds.insert(1, "bin_time", test_key["bin_time"].to_numpy())
        preds.to_parquet(out_root / "preds.parquet", index=False)


def _run_benchmark(cfg_path: str) -> None:
    cfg0 = load_experiment_config(cfg_path)
    events = load_event_table(cfg0.dataset)
    patient_index = make_patient_index(events, cfg0.dataset.time_col, cfg0.dataset.patient_id_col)
    splits = make_splits(patient_index, cfg0.split)

    binned = bin_events(events, cfg0.dataset, cfg0.preprocess).table

    labels_res = run_label_fn(events, cfg0)
    labels_df = None
    if labels_res is not None:
        if cfg0.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(labels_res.df)
        elif cfg0.task.prediction_mode == "time":
            labels_df = normalize_time_labels(labels_res.df, cfg0)

    if cfg0.task.prediction_mode == "patient":
        X, y0 = make_patient_tabular(binned)
        if labels_df is not None:
            y = labels_df.set_index("patient_id")["label"].reindex(X.index.astype(str))
        else:
            y = y0
        key = None
        global_mask = None
    elif cfg0.task.prediction_mode == "time":
        X0, y0, key0 = make_time_tabular(binned)
        if labels_df is not None:
            feat = key0.copy()
            feat["_row"] = feat.index
            lab = labels_df.merge(feat, on=["patient_id", "bin_time"], how="inner")
            X = X0.loc[lab["_row"].to_numpy()].reset_index(drop=True)
            y = lab["label"].reset_index(drop=True)
            key = lab[["patient_id", "bin_time"]].reset_index(drop=True)
            global_mask = lab["mask"].to_numpy().astype(bool)
        else:
            X, y, key = X0, y0, key0
            global_mask = None
    else:
        raise SystemExit(f"Unsupported task.prediction_mode={cfg0.task.prediction_mode!r}")

    import pandas as pd

    if cfg0.model.name in {"gru", "rnn", "transformer"}:
        torch = optional_import("torch")
        if torch is None:
            raise SystemExit(
                "Missing optional dependency: torch. Install it first (e.g. `uv add torch`)."
            )

    rows = []
    out_root = cfg0.output.root / cfg0.output.run_name
    ensure_dir(out_root)

    # Single-process only.

    for sp in splits:
        test_key = None

        # Select best hyperparameters using validation.
        def _eval_trial(cfg) -> tuple[float, dict[str, float]] | None:
            if cfg.task.prediction_mode == "patient":
                train_mask = X.index.astype(str).isin(sp.train_patients)
                val_mask = X.index.astype(str).isin(sp.val_patients)
                test_mask = X.index.astype(str).isin(sp.test_patients)
            else:
                assert key is not None
                train_mask = key["patient_id"].astype(str).isin(sp.train_patients)
                val_mask = key["patient_id"].astype(str).isin(sp.val_patients)
                test_mask = key["patient_id"].astype(str).isin(sp.test_patients)

            if global_mask is not None:
                train_mask2 = train_mask & global_mask
                val_mask2 = val_mask & global_mask
            else:
                train_mask2 = train_mask
                val_mask2 = val_mask

            X_train, y_train = X.loc[train_mask2], y.loc[train_mask2].to_numpy()
            X_val, y_val = X.loc[val_mask2], y.loc[val_mask2].to_numpy()

            # If users configured HPO metric incompatible with current trial type,
            # fall back to the trainer monitor for DL trials.
            hpo_metric = cfg.hpo.metric

            if cfg.model.name == "xgboost":
                if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                    return None
                art = train_xgboost(X_train, y_train, X_val, y_val, cfg.task, cfg.model.xgboost)
                y_val_score = predict_xgboost(art, X_val, cfg.task)
                if cfg.task.kind == "binary":
                    vm = binary_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
                    if hpo_metric in {"val_auroc", "auroc"}:
                        return float(vm["auroc"]), vm
                    return float(vm["auprc"]), vm
                vm = regression_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
                if hpo_metric in {"val_rmse", "rmse"}:
                    return float(vm["rmse"]), vm
                return float(vm["mae"]), vm

            torch = optional_import("torch")
            if torch is None:
                raise SystemExit("DL HPO requires torch")

            from oneehr.data.sequence import build_patient_sequences, build_time_sequences, pad_sequences

            feat_cols = [
                c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")
            ]

            if cfg.task.prediction_mode == "patient":
                pids, seqs, lens = build_patient_sequences(binned, feat_cols)
                X_seq = pad_sequences(seqs, lens)
                lens_t = torch.from_numpy(lens)

                y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
                y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)
                pids_arr = np.array(pids, dtype=str)

                tr_m = np.isin(pids_arr, sp.train_patients)
                va_m = np.isin(pids_arr, sp.val_patients)
                X_tr, L_tr, y_tr = X_seq[tr_m], lens_t[tr_m], torch.from_numpy(y_arr[tr_m])
                X_va, L_va, y_va = X_seq[va_m], lens_t[va_m], torch.from_numpy(y_arr[va_m])

                input_dim = int(X_seq.shape[-1])
                if cfg.model.name == "gru":
                    from oneehr.models.gru import GRUModel

                    model = GRUModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.gru.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.gru.num_layers,
                        dropout=cfg.model.gru.dropout,
                    )
                elif cfg.model.name == "rnn":
                    from oneehr.models.rnn import RNNModel

                    model = RNNModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.rnn.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.rnn.num_layers,
                        dropout=cfg.model.rnn.dropout,
                        bidirectional=cfg.model.rnn.bidirectional,
                        nonlinearity=cfg.model.rnn.nonlinearity,
                    )
                elif cfg.model.name == "transformer":
                    from oneehr.models.transformer import TransformerModel

                    model = TransformerModel(
                        input_dim=input_dim,
                        d_model=cfg.model.transformer.d_model,
                        out_dim=1,
                        nhead=cfg.model.transformer.nhead,
                        num_layers=cfg.model.transformer.num_layers,
                        dim_feedforward=cfg.model.transformer.dim_feedforward,
                        dropout=cfg.model.transformer.dropout,
                        pooling=cfg.model.transformer.pooling,
                    )
                else:
                    return None

                fit = fit_sequence_model(
                    model=model,
                    X_train=X_tr,
                    len_train=L_tr,
                    y_train=y_tr,
                    X_val=X_va,
                    len_val=L_va,
                    y_val=y_va,
                    task=cfg.task,
                    trainer=cfg.trainer,
                )
                last = fit.history[-1] if fit.history else {}
                monitor = cfg.trainer.monitor
                score = float(last.get(monitor, last.get("val_loss", 0.0)))
                return score, {monitor: score}

            if cfg.task.prediction_mode == "time":
                if labels_df is None:
                    raise SystemExit("prediction_mode='time' requires labels (labels.fn).")

                pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
                    binned,
                    labels_df,
                    feat_cols,
                    label_time_col="bin_time",
                )
                X_seq = pad_sequences(seqs, lens)
                Y_seq = pad_sequences([y[:, None] for y in y_seqs], lens).squeeze(-1)
                M_seq = pad_sequences([m[:, None] for m in mask_seqs], lens).squeeze(-1)
                lens_t = torch.from_numpy(lens)

                pids_arr = np.array(pids, dtype=str)
                tr_m = np.isin(pids_arr, sp.train_patients)
                va_m = np.isin(pids_arr, sp.val_patients)
                X_tr, L_tr, Y_tr, M_tr = (
                    X_seq[tr_m],
                    lens_t[tr_m],
                    Y_seq[tr_m],
                    M_seq[tr_m],
                )
                X_va, L_va, Y_va, M_va = (
                    X_seq[va_m],
                    lens_t[va_m],
                    Y_seq[va_m],
                    M_seq[va_m],
                )

                input_dim = int(X_seq.shape[-1])
                if cfg.model.name == "gru":
                    from oneehr.models.gru import GRUTimeModel

                    model = GRUTimeModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.gru.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.gru.num_layers,
                        dropout=cfg.model.gru.dropout,
                    )
                elif cfg.model.name == "rnn":
                    from oneehr.models.rnn import RNNTimeModel

                    model = RNNTimeModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.rnn.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.rnn.num_layers,
                        dropout=cfg.model.rnn.dropout,
                        bidirectional=cfg.model.rnn.bidirectional,
                        nonlinearity=cfg.model.rnn.nonlinearity,
                    )
                elif cfg.model.name == "transformer":
                    from oneehr.models.transformer import TransformerTimeModel

                    model = TransformerTimeModel(
                        input_dim=input_dim,
                        d_model=cfg.model.transformer.d_model,
                        out_dim=1,
                        nhead=cfg.model.transformer.nhead,
                        num_layers=cfg.model.transformer.num_layers,
                        dim_feedforward=cfg.model.transformer.dim_feedforward,
                        dropout=cfg.model.transformer.dropout,
                    )
                else:
                    return None

                fit = fit_sequence_model_time(
                    model=model,
                    X_train=X_tr,
                    len_train=L_tr,
                    y_train=Y_tr,
                    mask_train=M_tr,
                    X_val=X_va,
                    len_val=L_va,
                    y_val=Y_va,
                    mask_val=M_va,
                    task=cfg.task,
                    trainer=cfg.trainer,
                )
                last = fit.history[-1] if fit.history else {}
                monitor = cfg.trainer.monitor
                score = float(last.get(monitor, last.get("val_loss", 0.0)))
                return score, {monitor: score}

            return None

        hpo_res = select_best_with_trials(cfg0, _eval_trial)
        best_overrides = hpo_res.best.overrides if hpo_res.best is not None else {}

        # Persist trial results per split.
        trial_rows = []
        for tr in hpo_res.trials:
            trial_rows.append(
                {
                    "split": sp.name,
                    "hpo_metric": cfg0.hpo.metric,
                    "hpo_mode": cfg0.hpo.mode,
                    "trial_score": tr.score,
                    "overrides": str(tr.overrides),
                    **{f"trial_{k}": v for k, v in tr.metrics.items()},
                }
            )
        import pandas as pd

        ensure_dir(out_root / "hpo")
        pd.DataFrame(trial_rows).to_csv(out_root / "hpo" / f"trials_{sp.name}.csv", index=False)
        write_json(
            out_root / "hpo" / f"best_{sp.name}.json",
            {
                "split": sp.name,
                "metric": cfg0.hpo.metric,
                "mode": cfg0.hpo.mode,
                "best": None if hpo_res.best is None else {
                    "score": hpo_res.best.score,
                    "overrides": hpo_res.best.overrides,
                    "metrics": hpo_res.best.metrics,
                },
            },
        )

        cfg = apply_overrides(cfg0, best_overrides)

        # Train on this split using the selected hyperparameters, then evaluate on test.
        if cfg.model.name == "xgboost":
            if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                # Keep a row for completeness, but mark as skipped.
                row = {"split": sp.name, "skipped": 1, "reason": "single_class_train"}
                rows.append(row)
                continue

            art = train_xgboost(X_train, y_train, X_val, y_val, cfg.task, cfg.model.xgboost)
            y_score = predict_xgboost(art, X_test, cfg.task)
        else:
            input_dim = len(
                [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
            )
            if cfg.model.name == "gru":
                from oneehr.models.gru import GRUModel, GRUTimeModel

                if cfg.task.prediction_mode == "patient":
                    model = GRUModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.gru.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.gru.num_layers,
                        dropout=cfg.model.gru.dropout,
                    )
                    y_score, y_test, test_patient_ids = _train_sequence_patient_level(
                        model=model,
                        binned=binned,
                        y=y,
                        split=sp,
                        cfg=cfg.trainer,
                        task=cfg.task,
                    )
                else:
                    model = GRUTimeModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.gru.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.gru.num_layers,
                        dropout=cfg.model.gru.dropout,
                    )
                    y_score, y_test, test_key_rows = _train_sequence_time_level(
                        model=model,
                        binned=binned,
                        labels_df=labels_df,
                        split=sp,
                        cfg=cfg.trainer,
                        task=cfg.task,
                    )
                    test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                    test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])
            elif cfg.model.name == "rnn":
                from oneehr.models.rnn import RNNModel, RNNTimeModel

                if cfg.task.prediction_mode == "patient":
                    model = RNNModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.rnn.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.rnn.num_layers,
                        dropout=cfg.model.rnn.dropout,
                        bidirectional=cfg.model.rnn.bidirectional,
                        nonlinearity=cfg.model.rnn.nonlinearity,
                    )
                    y_score, y_test, test_patient_ids = _train_sequence_patient_level(
                        model=model,
                        binned=binned,
                        y=y,
                        split=sp,
                        cfg=cfg.trainer,
                        task=cfg.task,
                    )
                else:
                    model = RNNTimeModel(
                        input_dim=input_dim,
                        hidden_dim=cfg.model.rnn.hidden_dim,
                        out_dim=1,
                        num_layers=cfg.model.rnn.num_layers,
                        dropout=cfg.model.rnn.dropout,
                        bidirectional=cfg.model.rnn.bidirectional,
                        nonlinearity=cfg.model.rnn.nonlinearity,
                    )
                    y_score, y_test, test_key_rows = _train_sequence_time_level(
                        model=model,
                        binned=binned,
                        labels_df=labels_df,
                        split=sp,
                        cfg=cfg.trainer,
                        task=cfg.task,
                    )
                    test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                    test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])
            else:
                from oneehr.models.transformer import TransformerModel, TransformerTimeModel

                if cfg.task.prediction_mode == "patient":
                    model = TransformerModel(
                        input_dim=input_dim,
                        d_model=cfg.model.transformer.d_model,
                        out_dim=1,
                        nhead=cfg.model.transformer.nhead,
                        num_layers=cfg.model.transformer.num_layers,
                        dim_feedforward=cfg.model.transformer.dim_feedforward,
                        dropout=cfg.model.transformer.dropout,
                        pooling=cfg.model.transformer.pooling,
                    )
                    y_score, y_test, test_patient_ids = _train_sequence_patient_level(
                        model=model,
                        binned=binned,
                        y=y,
                        split=sp,
                        cfg=cfg.trainer,
                        task=cfg.task,
                    )
                else:
                    model = TransformerTimeModel(
                        input_dim=input_dim,
                        d_model=cfg.model.transformer.d_model,
                        out_dim=1,
                        nhead=cfg.model.transformer.nhead,
                        num_layers=cfg.model.transformer.num_layers,
                        dim_feedforward=cfg.model.transformer.dim_feedforward,
                        dropout=cfg.model.transformer.dropout,
                    )
                    y_score, y_test, test_key_rows = _train_sequence_time_level(
                        model=model,
                        binned=binned,
                        labels_df=labels_df,
                        split=sp,
                        cfg=cfg.trainer,
                        task=cfg.task,
                    )
                    test_patient_ids = np.array([r[0] for r in test_key_rows], dtype=str)
                    test_key = pd.DataFrame(test_key_rows, columns=["patient_id", "bin_time"])

        if cfg.task.kind == "binary":
            metrics = binary_metrics(y_test.astype(float), y_score.astype(float)).metrics
        else:
            metrics = regression_metrics(y_test.astype(float), y_score.astype(float)).metrics

        row = {
            "split": sp.name,
            "skipped": 0,
            **metrics,
            "hpo_metric": cfg0.hpo.metric,
            "hpo_mode": cfg0.hpo.mode,
            "hpo_best_score": None if hpo_res.best is None else hpo_res.best.score,
            "hpo_best": str(best_overrides),
        }
        rows.append(row)

        if cfg.output.save_preds and len(test_patient_ids) > 0:
            preds = pd.DataFrame(
                {"patient_id": test_patient_ids, "y_true": y_test, "y_pred": y_score}
            )
            if test_key is not None:
                preds.insert(1, "bin_time", test_key["bin_time"].to_numpy())
            ensure_dir(out_root / "preds")
            preds.to_parquet(out_root / "preds" / f"{sp.name}.parquet", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "summary.csv", index=False)
    summarize_metrics(summary).to_csv(out_root / "summary_table.csv", index=False)
    to_paper_wide_table(summary).to_csv(out_root / "paper_table.csv", index=False)

    # Persist best overrides per split for reproducibility.
    best_rows = [{"split": r["split"], "hpo_best": r.get("hpo_best")} for r in rows]
    pd.DataFrame(best_rows).to_csv(out_root / "hpo_best.csv", index=False)


def _run_hpo(cfg_path: str) -> None:
    """Run a pure HPO selection on the first split (validation-driven).

    Intended for quick iteration; `benchmark` already runs per-split selection.
    """

    cfg0 = load_experiment_config(cfg_path)
    events = load_event_table(cfg0.dataset)
    patient_index = make_patient_index(events, cfg0.dataset.time_col, cfg0.dataset.patient_id_col)
    splits = make_splits(patient_index, cfg0.split)
    sp = splits[0]

    binned = bin_events(events, cfg0.dataset, cfg0.preprocess).table

    labels_res = run_label_fn(events, cfg0)
    labels_df = None
    if labels_res is not None:
        if cfg0.task.prediction_mode == "patient":
            labels_df = normalize_patient_labels(labels_res.df)
        else:
            labels_df = normalize_time_labels(labels_res.df, cfg0)

    if cfg0.task.prediction_mode == "patient":
        X, y0 = make_patient_tabular(binned)
        if labels_df is not None:
            y = labels_df.set_index("patient_id")["label"].reindex(X.index.astype(str))
        else:
            y = y0
        train_mask = X.index.astype(str).isin(sp.train_patients)
        val_mask = X.index.astype(str).isin(sp.val_patients)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask].to_numpy()
        X_val, y_val = X.loc[val_mask], y.loc[val_mask].to_numpy()
    else:
        raise SystemExit("hpo currently supports prediction_mode='patient' only")

    best_overrides = None
    best_val = None

    def _better(mode: str, a: float, b: float) -> bool:
        return a < b if mode == "min" else a > b

    for overrides in iter_grid(cfg0.hpo):
        cfg = apply_overrides(cfg0, overrides)
        if cfg.model.name != "xgboost":
            continue
        if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
            continue
        art = train_xgboost(X_train, y_train, X_val, y_val, cfg.task, cfg.model.xgboost)
        y_val_score = predict_xgboost(art, X_val, cfg.task)
        if cfg.task.kind == "binary":
            vm = binary_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
            val_score = float(vm["auroc"]) if cfg.hpo.metric in {"val_auroc", "auroc"} else float(vm["auprc"])
        else:
            vm = regression_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
            val_score = float(vm["rmse"]) if cfg.hpo.metric in {"val_rmse", "rmse"} else float(vm["mae"])

        if best_val is None or _better(cfg.hpo.mode, val_score, best_val):
            best_val = val_score
            best_overrides = dict(overrides)

    out_root = cfg0.output.root / cfg0.output.run_name
    ensure_dir(out_root)
    write_json(out_root / "hpo_best.json", {"best_overrides": best_overrides, "best_val": best_val})


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        _run_preprocess(args.config)
        return
    if args.command == "train":
        _run_train(args.config)
        return
    if args.command == "benchmark":
        _run_benchmark(args.config)
        return
    if args.command == "hpo":
        _run_hpo(args.config)
        return
    raise SystemExit(f"Unknown command: {args.command}")
