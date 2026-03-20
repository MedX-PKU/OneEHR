"""HPO scope handlers for the train pipeline."""
from __future__ import annotations

import sys

import numpy as np

from oneehr.utils.io import ensure_dir, write_json


def run_single_hpo(
    cfg_model,
    model_name: str,
    X, y, key, global_mask,
    splits,
    out_root,
    *,
    binned=None,
    labels_df=None,
    dynamic_feature_columns=None,
):
    """Run HPO with scope='single' — evaluate on a single tune split.

    Returns the best overrides dict (empty if no best found).
    """
    import torch
    from oneehr.hpo.grid import iter_grid
    from oneehr.hpo.runner import select_best_with_trials
    from oneehr.models.tree import predict_tabular, train_tabular_model
    from oneehr.eval.metrics import binary_metrics, regression_metrics

    if cfg_model.hpo.tune_split is not None:
        tune_splits = [sp for sp in splits if sp.name == cfg_model.hpo.tune_split]
        if not tune_splits:
            raise SystemExit(
                f"hpo.tune_split={cfg_model.hpo.tune_split!r} not found in splits: "
                f"{[sp.name for sp in splits]}"
            )
        tune_split = tune_splits[0]
    else:
        tune_split = splits[0]

    tr_m, va_m = _make_masks(cfg_model, X, key, global_mask, tune_split)
    X_train_h, y_train_h = X.loc[tr_m], y.loc[tr_m].to_numpy()
    X_val_h, y_val_h = X.loc[va_m], y.loc[va_m].to_numpy()

    def _eval_trial_once(cfg) -> tuple[float, dict[str, float]] | None:
        hpo_metric = cfg.hpo.metric
        if cfg.model.name != "xgboost":
            return None
        if cfg.task.kind == "binary" and len(np.unique(y_train_h)) < 2:
            return None
        art = train_tabular_model(
            model_name="xgboost",
            X_train=X_train_h, y_train=y_train_h,
            X_val=X_val_h, y_val=y_val_h,
            task=cfg.task, model_cfg=cfg.model.xgboost,
        )
        y_val_score = predict_tabular(art, X_val_h, cfg.task)
        if cfg.task.kind == "binary":
            vm = binary_metrics(y_val_h.astype(float), y_val_score.astype(float)).metrics
            if hpo_metric in {"val_auroc", "auroc"}:
                return float(vm["auroc"]), vm
            return float(vm["auprc"]), vm
        vm = regression_metrics(y_val_h.astype(float), y_val_score.astype(float)).metrics
        if hpo_metric in {"val_rmse", "rmse"}:
            return float(vm["rmse"]), vm
        return float(vm["mae"]), vm

    hpo_res_once = select_best_with_trials(cfg_model, _eval_trial_once)
    overrides = hpo_res_once.best.overrides if hpo_res_once.best is not None else {}
    hpo_dir = ensure_dir(out_root / "hpo" / model_name)
    write_json(
        hpo_dir / "best_once.json",
        {
            "split": tune_split.name,
            "metric": cfg_model.hpo.metric,
            "mode": cfg_model.hpo.mode,
            "scope": cfg_model.hpo.scope,
            "best": None if hpo_res_once.best is None else {
                "score": hpo_res_once.best.score,
                "overrides": hpo_res_once.best.overrides,
                "metrics": hpo_res_once.best.metrics,
            },
        },
    )
    return overrides


def run_cv_mean_hpo(
    cfg_model,
    model_name: str,
    X, y, key, global_mask,
    splits,
    out_root,
):
    """Run HPO with scope='cv_mean' — average across all splits.

    Returns the best overrides dict (empty if no best found).
    """
    from oneehr.hpo.grid import apply_overrides, iter_grid
    from oneehr.models.tree import predict_tabular, train_tabular_model
    from oneehr.eval.metrics import binary_metrics, regression_metrics

    trials = []
    best = None

    def _better(a: float, b: float) -> bool:
        return a < b if cfg_model.hpo.mode == "min" else a > b

    for overrides in iter_grid(cfg_model.hpo):
        cfg_trial = apply_overrides(cfg_model, overrides)
        split_scores = []

        for sp in splits:
            tr_m, va_m = _make_masks(cfg_model, X, key, global_mask, sp)
            X_train_s, y_train_s = X.loc[tr_m], y.loc[tr_m].to_numpy()
            X_val_s, y_val_s = X.loc[va_m], y.loc[va_m].to_numpy()

            if cfg_trial.model.name != "xgboost":
                continue
            if cfg_trial.task.kind == "binary" and len(np.unique(y_train_s)) < 2:
                continue
            art = train_tabular_model(
                model_name="xgboost",
                X_train=X_train_s, y_train=y_train_s,
                X_val=X_val_s, y_val=y_val_s,
                task=cfg_trial.task, model_cfg=cfg_trial.model.xgboost,
            )
            y_val_score = predict_tabular(art, X_val_s, cfg_trial.task)

            if cfg_trial.task.kind == "binary":
                vm = binary_metrics(y_val_s.astype(float), y_val_score.astype(float)).metrics
                if cfg_trial.hpo.aggregate_metric is not None:
                    if cfg_trial.hpo.aggregate_metric not in vm:
                        raise SystemExit(
                            f"hpo.aggregate_metric={cfg_trial.hpo.aggregate_metric!r} not in metrics: {list(vm.keys())}"
                        )
                    score = float(vm[cfg_trial.hpo.aggregate_metric])
                else:
                    score = float(vm["auroc"]) if cfg_trial.hpo.metric in {"val_auroc", "auroc"} else float(vm["auprc"])
            else:
                vm = regression_metrics(y_val_s.astype(float), y_val_score.astype(float)).metrics
                if cfg_trial.hpo.aggregate_metric is not None:
                    if cfg_trial.hpo.aggregate_metric not in vm:
                        raise SystemExit(
                            f"hpo.aggregate_metric={cfg_trial.hpo.aggregate_metric!r} not in metrics: {list(vm.keys())}"
                        )
                    score = float(vm[cfg_trial.hpo.aggregate_metric])
                else:
                    score = float(vm["rmse"]) if cfg_trial.hpo.metric in {"val_rmse", "rmse"} else float(vm["mae"])

            split_scores.append({"split": sp.name, "score": score})

        if not split_scores:
            continue

        mean_score = float(np.mean([r["score"] for r in split_scores]))
        trial = {"mean_score": mean_score, "overrides": dict(overrides), "per_split": split_scores}
        trials.append(trial)

        if best is None or _better(mean_score, best["mean_score"]):
            best = trial

    hpo_dir = ensure_dir(out_root / "hpo" / model_name)
    write_json(
        hpo_dir / "select_best_split.json",
        {
            "metric": cfg_model.hpo.metric,
            "mode": cfg_model.hpo.mode,
            "selected_mean_score": None if best is None else best["mean_score"],
            "selected_overrides": None if best is None else best["overrides"],
            "trials": trials,
        },
    )
    return {} if best is None else best["overrides"]


def run_per_split_hpo(
    cfg_model,
    model_name: str,
    sp,
    X, y, key, global_mask,
    out_root,
    *,
    X_train, y_train, X_val, y_val,
    binned=None,
    labels_df=None,
    dynamic_feature_columns=None,
):
    """Run HPO with scope='per_split' — evaluate on this split only.

    Returns ``(best_overrides, hpo_best_score)``.
    """
    import pandas as pd
    import torch
    from dataclasses import replace

    from oneehr.hpo.runner import select_best_with_trials
    from oneehr.hpo.grid import iter_grid
    from oneehr.models.tree import predict_tabular, train_tabular_model
    from oneehr.models import build_model
    from oneehr.modeling.trainer import fit_model
    from oneehr.eval.metrics import binary_metrics, regression_metrics
    from oneehr.models import TABULAR_MODELS

    def _eval_trial(cfg) -> tuple[float, dict[str, float]] | None:
        hpo_metric = cfg.hpo.metric

        if cfg.model.name in TABULAR_MODELS:
            if cfg.task.kind == "binary" and len(np.unique(y_train)) < 2:
                return None
            tab_model_cfg = getattr(cfg.model, cfg.model.name)
            art = train_tabular_model(
                model_name=cfg.model.name,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                task=cfg.task, model_cfg=tab_model_cfg,
            )
            y_val_score = predict_tabular(art, X_val, cfg.task)
            if cfg.task.kind == "binary":
                vm = binary_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
                if hpo_metric in {"val_auroc", "auroc"}:
                    return float(vm["auroc"]), vm
                return float(vm["auprc"]), vm
            vm = regression_metrics(y_val.astype(float), y_val_score.astype(float)).metrics
            if hpo_metric in {"val_rmse", "rmse"}:
                return float(vm["rmse"]), vm
            return float(vm["mae"]), vm

        if dynamic_feature_columns is None or len(dynamic_feature_columns) == 0:
            return None

        from oneehr.data.sequence import build_patient_sequences, build_time_sequences, pad_sequences

        feat_cols = list(dynamic_feature_columns)

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
            X_va, L_va, y_va_t = X_seq[va_m], lens_t[va_m], torch.from_numpy(y_arr[va_m])

            input_dim = int(X_seq.shape[-1])
            cfg_use = replace(cfg, _dynamic_dim=input_dim)
            built = build_model(cfg_use)
            if built.kind != "dl":
                return None
            mdl = built.model

            fit = fit_model(
                model=mdl,
                X_train=X_tr, len_train=L_tr, y_train=y_tr, static_train=None,
                X_val=X_va, len_val=L_va, y_val=y_va_t, static_val=None,
                task=cfg.task, trainer=cfg.trainer,
            )
            last = fit.history[-1] if fit.history else {}
            monitor = cfg.trainer.monitor
            score = float(last.get(monitor, last.get("val_loss", 0.0)))
            return score, {monitor: score}

        if cfg.task.prediction_mode == "time":
            if labels_df is None:
                raise SystemExit("prediction_mode='time' requires labels (labels.fn).")

            pids, time_seqs, seqs, y_seqs, mask_seqs, lens = build_time_sequences(
                binned, labels_df, feat_cols, label_time_col="bin_time",
            )
            X_seq = pad_sequences(seqs, lens)
            Y_seq = pad_sequences([yy[:, None] for yy in y_seqs], lens).squeeze(-1)
            M_seq = pad_sequences([mm[:, None] for mm in mask_seqs], lens).squeeze(-1)
            lens_t = torch.from_numpy(lens)

            pids_arr = np.array(pids, dtype=str)
            tr_m = np.isin(pids_arr, sp.train_patients)
            va_m = np.isin(pids_arr, sp.val_patients)
            X_tr, L_tr, Y_tr, M_tr = X_seq[tr_m], lens_t[tr_m], Y_seq[tr_m], M_seq[tr_m]
            X_va, L_va, Y_va, M_va = X_seq[va_m], lens_t[va_m], Y_seq[va_m], M_seq[va_m]

            input_dim = int(X_seq.shape[-1])
            cfg_use = replace(cfg, _dynamic_dim=input_dim)
            built = build_model(cfg_use)
            if built.kind != "dl":
                return None
            mdl = built.model

            fit = fit_model(
                model=mdl,
                X_train=X_tr, len_train=L_tr, y_train=Y_tr, static_train=None,
                X_val=X_va, len_val=L_va, y_val=Y_va, static_val=None,
                task=cfg.task, trainer=cfg.trainer,
                mask_train=M_tr, mask_val=M_va,
            )
            last = fit.history[-1] if fit.history else {}
            monitor = cfg.trainer.monitor
            score = float(last.get(monitor, last.get("val_loss", 0.0)))
            return score, {monitor: score}

        return None

    hpo_res = select_best_with_trials(cfg_model, _eval_trial)
    best_overrides = hpo_res.best.overrides if hpo_res.best is not None else {}

    trial_rows = []
    for tr in hpo_res.trials:
        trial_rows.append(
            {
                "model": model_name,
                "split": sp.name,
                "hpo_metric": cfg_model.hpo.metric,
                "hpo_mode": cfg_model.hpo.mode,
                "trial_score": tr.score,
                "overrides": str(tr.overrides),
                **{f"trial_{k}": v for k, v in tr.metrics.items()},
            }
        )

    hpo_dir = ensure_dir(out_root / "hpo" / model_name)
    pd.DataFrame(trial_rows).to_csv(hpo_dir / f"trials_{sp.name}.csv", index=False)
    write_json(
        hpo_dir / f"best_{sp.name}.json",
        {
            "split": sp.name,
            "metric": cfg_model.hpo.metric,
            "mode": cfg_model.hpo.mode,
            "best": None if hpo_res.best is None else {
                "score": hpo_res.best.score,
                "overrides": hpo_res.best.overrides,
                "metrics": hpo_res.best.metrics,
            },
        },
    )
    hpo_best_score = None if hpo_res.best is None else hpo_res.best.score
    return best_overrides, hpo_best_score


def _make_masks(cfg_model, X, key, global_mask, sp):
    """Return ``(train_mask, val_mask)`` boolean Series aligned on X."""
    if cfg_model.task.prediction_mode == "patient":
        tr = X.index.astype(str).isin(sp.train_patients)
        va = X.index.astype(str).isin(sp.val_patients)
    else:
        assert key is not None
        tr = key["patient_id"].astype(str).isin(sp.train_patients)
        va = key["patient_id"].astype(str).isin(sp.val_patients)
    if global_mask is not None:
        tr = tr & global_mask
        va = va & global_mask
    return tr, va
