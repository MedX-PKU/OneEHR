import argparse

from oneehr.config.load import load_experiment_config
from oneehr.core.index import make_patient_index
from oneehr.data.io import load_event_table
from oneehr.featurize.tabular import make_patient_tabular, make_time_tabular
from oneehr.models.xgb import predict_xgboost, train_xgboost
from oneehr.preprocess import bin_events
from oneehr.task.align import normalize_patient_labels, normalize_time_labels
from oneehr.task.label_fn import run_label_fn
from oneehr.training.gru_trainer import train_gru_patient
from oneehr.training.metrics import binary_metrics, regression_metrics
from oneehr.training.sequence import build_patient_sequences, pad_sequences
from oneehr.training.splits import make_splits


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oneehr")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Preprocess raw event table into model-ready artifacts")
    p_pre.add_argument("--config", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True)

    p_bm = sub.add_parser("benchmark", help="Run benchmarking across splits/models")
    p_bm.add_argument("--config", required=True)

    return parser


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def _run_preprocess(cfg_path: str) -> None:
    cfg = load_experiment_config(cfg_path)
    events = load_event_table(cfg.dataset)
    binned = bin_events(events, cfg.dataset, cfg.preprocess)

    labels_res = run_label_fn(events, cfg)

    out_root = cfg.output.root / cfg.output.run_name
    _ensure_dir(out_root)
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

    if cfg.model.name != "xgboost":
        if cfg.model.name != "gru":
            raise SystemExit(f"Unsupported model.name={cfg.model.name!r}")
        if cfg.task.prediction_mode != "patient":
            raise SystemExit("MVP: GRU currently supports prediction_mode='patient' only")

        feature_cols = list(X_train.columns)
        # Rebuild sequences from binned table (patient-level) using same features.
        feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
        pids, seqs, lens = build_patient_sequences(binned, feat_cols)
        X_seq = pad_sequences(seqs, lens)
        lens_t = torch.from_numpy(lens)
        y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
        y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)

        # Align to split0.
        pids_arr = np.array(pids, dtype=str)
        train_m = np.isin(pids_arr, split0.train_patients)
        val_m = np.isin(pids_arr, split0.val_patients)
        test_m = np.isin(pids_arr, split0.test_patients)

        X_tr, L_tr, y_tr = X_seq[train_m], lens_t[train_m], torch.from_numpy(y_arr[train_m])
        X_va, L_va, y_va = X_seq[val_m], lens_t[val_m], torch.from_numpy(y_arr[val_m])
        X_te, L_te, y_te = X_seq[test_m], lens_t[test_m], torch.from_numpy(y_arr[test_m])

        _art, val_preds = train_gru_patient(
            feature_columns=feat_cols,
            X_train=X_tr,
            len_train=L_tr,
            y_train=y_tr,
            X_val=X_te,  # evaluate on test for now
            len_val=L_te,
            y_val=y_te,
            task=cfg.task,
            cfg=cfg.model.gru,
        )
        y_score = val_preds.y_pred
        y_test = val_preds.y_true
        test_patient_ids = pids_arr[test_m]
        test_key = None
    else:
        art = train_xgboost(X_train, y_train, X_val, y_val, cfg.task, cfg.model.xgboost)
        y_score = predict_xgboost(art, X_test, cfg.task)

    if cfg.task.kind == "binary":
        metrics = binary_metrics(y_test.astype(float), y_score.astype(float)).metrics
    else:
        metrics = regression_metrics(y_test.astype(float), y_score.astype(float)).metrics

    out_root = cfg.output.root / cfg.output.run_name
    _ensure_dir(out_root)
    (out_root / "metrics.json").write_text(str(metrics), encoding="utf-8")

    if cfg.output.save_preds:
        import pandas as pd

        preds = pd.DataFrame({"patient_id": test_patient_ids, "y_true": y_test, "y_pred": y_score})
        if test_key is not None:
            preds.insert(1, "bin_time", test_key["bin_time"].to_numpy())
        preds.to_parquet(out_root / "preds.parquet", index=False)


def _run_benchmark(cfg_path: str) -> None:
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

    if cfg.task.prediction_mode == "patient":
        X, y0 = make_patient_tabular(binned)
        if labels_df is not None:
            y = labels_df.set_index("patient_id")["label"].reindex(X.index.astype(str))
        else:
            y = y0
        key = None
        global_mask = None
    elif cfg.task.prediction_mode == "time":
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
        raise SystemExit(f"Unsupported task.prediction_mode={cfg.task.prediction_mode!r}")

    import pandas as pd

    rows = []
    out_root = cfg.output.root / cfg.output.run_name
    _ensure_dir(out_root)

    for sp in splits:
        if cfg.task.prediction_mode == "patient":
            train_mask = X.index.astype(str).isin(sp.train_patients)
            val_mask = X.index.astype(str).isin(sp.val_patients)
            test_mask = X.index.astype(str).isin(sp.test_patients)
            test_key = None
            test_patient_ids = X.loc[test_mask].index.astype(str)
        else:
            assert key is not None
            train_mask = key["patient_id"].astype(str).isin(sp.train_patients)
            val_mask = key["patient_id"].astype(str).isin(sp.val_patients)
            test_mask = key["patient_id"].astype(str).isin(sp.test_patients)
            test_key = key.loc[test_mask].reset_index(drop=True)
            test_patient_ids = test_key["patient_id"].astype(str)

        if global_mask is not None:
            train_mask = train_mask & global_mask
            val_mask = val_mask & global_mask
            test_mask = test_mask & global_mask

        X_train, y_train = X.loc[train_mask], y.loc[train_mask].to_numpy()
        X_val, y_val = X.loc[val_mask], y.loc[val_mask].to_numpy()
        X_test, y_test = X.loc[test_mask], y.loc[test_mask].to_numpy()

        if cfg.model.name == "xgboost":
            art = train_xgboost(X_train, y_train, X_val, y_val, cfg.task, cfg.model.xgboost)
            y_score = predict_xgboost(art, X_test, cfg.task)
        elif cfg.model.name == "gru":
            if cfg.task.prediction_mode != "patient":
                raise SystemExit("MVP: GRU benchmark supports prediction_mode='patient' only")
            feat_cols = [c for c in binned.columns if c.startswith("num__") or c.startswith("cat__")]
            pids, seqs, lens = build_patient_sequences(binned, feat_cols)
            X_seq = pad_sequences(seqs, lens)
            lens_t = torch.from_numpy(lens)
            y_map = dict(zip(y.index.astype(str).tolist(), y.to_numpy().tolist()))
            y_arr = np.array([y_map.get(pid) for pid in pids], dtype=np.float32)
            pids_arr = np.array(pids, dtype=str)
            train_m = np.isin(pids_arr, sp.train_patients)
            val_m = np.isin(pids_arr, sp.val_patients)
            test_m = np.isin(pids_arr, sp.test_patients)
            X_tr, L_tr, y_tr = X_seq[train_m], lens_t[train_m], torch.from_numpy(y_arr[train_m])
            X_va, L_va, y_va = X_seq[val_m], lens_t[val_m], torch.from_numpy(y_arr[val_m])
            X_te, L_te, y_te = X_seq[test_m], lens_t[test_m], torch.from_numpy(y_arr[test_m])
            _art, te_preds = train_gru_patient(
                feature_columns=feat_cols,
                X_train=X_tr,
                len_train=L_tr,
                y_train=y_tr,
                X_val=X_te,
                len_val=L_te,
                y_val=y_te,
                task=cfg.task,
                cfg=cfg.model.gru,
            )
            y_score = te_preds.y_pred
            y_test = te_preds.y_true
            test_patient_ids = pids_arr[test_m]
            test_key = None
        else:
            raise SystemExit(f"Unsupported model.name={cfg.model.name!r}")

        if cfg.task.kind == "binary":
            metrics = binary_metrics(y_test.astype(float), y_score.astype(float)).metrics
        else:
            metrics = regression_metrics(y_test.astype(float), y_score.astype(float)).metrics

        row = {"split": sp.name, **metrics}
        rows.append(row)

        if cfg.output.save_preds:
            preds = pd.DataFrame(
                {"patient_id": test_patient_ids, "y_true": y_test, "y_pred": y_score}
            )
            if test_key is not None:
                preds.insert(1, "bin_time", test_key["bin_time"].to_numpy())
            _ensure_dir(out_root / "preds")
            preds.to_parquet(out_root / "preds" / f"{sp.name}.parquet", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "summary.csv", index=False)


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
    raise SystemExit(f"Unknown command: {args.command}")
