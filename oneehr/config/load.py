from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from oneehr.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    HPOConfig,
    LabelsConfig,
    ModelConfig,
    OutputConfig,
    PreprocessConfig,
    SplitConfig,
    TaskConfig,
    TrainerConfig,
    GRUConfig,
    RNNConfig,
    TransformerConfig,
    XGBoostConfig,
)


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    return d[key]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    dataset_raw = _require(raw, "dataset")
    preprocess_raw = raw.get("preprocess", {})
    task_raw = _require(raw, "task")
    split_raw = _require(raw, "split")
    model_raw = _require(raw, "model")
    output_raw = raw.get("output", {})
    labels_raw = raw.get("labels", {})
    trainer_raw = raw.get("trainer", {})
    hpo_raw = raw.get("hpo", {})

    dataset = DatasetConfig(
        path=Path(_require(dataset_raw, "path")),
        file_type=dataset_raw.get("file_type", "csv"),
        patient_id_col=dataset_raw.get("patient_id_col", "patient_id"),
        time_col=dataset_raw.get("time_col", "event_time"),
        code_col=dataset_raw.get("code_col", "code"),
        value_col=dataset_raw.get("value_col", "value"),
        label_col=dataset_raw.get("label_col", "label"),
        time_format=dataset_raw.get("time_format"),
    )

    preprocess = PreprocessConfig(
        bin_size=preprocess_raw.get("bin_size", "1d"),
        numeric_strategy=preprocess_raw.get("numeric_strategy", "mean"),
        categorical_strategy=preprocess_raw.get("categorical_strategy", "count"),
        top_k_codes=int(preprocess_raw.get("top_k_codes", 500)),
        min_code_count=int(preprocess_raw.get("min_code_count", 1)),
    )

    task = TaskConfig(
        kind=_require(task_raw, "kind"),
        prediction_mode=task_raw.get("prediction_mode", "patient"),
    )

    labels = LabelsConfig(
        fn=labels_raw.get("fn") or None,
        time_col=labels_raw.get("time_col", "label_time"),
        bin_from_time_col=bool(labels_raw.get("bin_from_time_col", True)),
    )

    split = SplitConfig(
        kind=_require(split_raw, "kind"),
        seed=int(split_raw.get("seed", 42)),
        n_splits=int(split_raw.get("n_splits", 5)),
        val_size=float(split_raw.get("val_size", 0.1)),
        test_size=float(split_raw.get("test_size", 0.2)),
        time_boundary=split_raw.get("time_boundary"),
    )

    xgb_raw = model_raw.get("xgboost", {})
    gru_raw = model_raw.get("gru", {})
    rnn_raw = model_raw.get("rnn", {})
    tf_raw = model_raw.get("transformer", {})
    model = ModelConfig(
        name=_require(model_raw, "name"),
        xgboost=XGBoostConfig(
            max_depth=int(xgb_raw.get("max_depth", 6)),
            n_estimators=int(xgb_raw.get("n_estimators", 500)),
            learning_rate=float(xgb_raw.get("learning_rate", 0.05)),
            subsample=float(xgb_raw.get("subsample", 0.8)),
            colsample_bytree=float(xgb_raw.get("colsample_bytree", 0.8)),
            reg_lambda=float(xgb_raw.get("reg_lambda", 1.0)),
            min_child_weight=float(xgb_raw.get("min_child_weight", 1.0)),
        ),
        gru=GRUConfig(
            hidden_dim=int(gru_raw.get("hidden_dim", 128)),
            num_layers=int(gru_raw.get("num_layers", 1)),
            dropout=float(gru_raw.get("dropout", 0.0)),
        ),
        rnn=RNNConfig(
            hidden_dim=int(rnn_raw.get("hidden_dim", 128)),
            num_layers=int(rnn_raw.get("num_layers", 1)),
            dropout=float(rnn_raw.get("dropout", 0.0)),
            bidirectional=bool(rnn_raw.get("bidirectional", False)),
            nonlinearity=str(rnn_raw.get("nonlinearity", "tanh")),
        ),
        transformer=TransformerConfig(
            d_model=int(tf_raw.get("d_model", 128)),
            nhead=int(tf_raw.get("nhead", 4)),
            num_layers=int(tf_raw.get("num_layers", 2)),
            dim_feedforward=int(tf_raw.get("dim_feedforward", 256)),
            dropout=float(tf_raw.get("dropout", 0.1)),
            pooling=str(tf_raw.get("pooling", "last")),
        ),
    )

    output = OutputConfig(
        root=Path(output_raw.get("root", "outputs")),
        run_name=str(output_raw.get("run_name", "run")),
        save_preds=bool(output_raw.get("save_preds", True)),
    )

    trainer = TrainerConfig(
        device=str(trainer_raw.get("device", "auto")),
        precision=str(trainer_raw.get("precision", "fp32")),
        seed=int(trainer_raw.get("seed", 42)),
        max_epochs=int(trainer_raw.get("max_epochs", 30)),
        batch_size=int(trainer_raw.get("batch_size", 64)),
        lr=float(trainer_raw.get("lr", 1e-3)),
        weight_decay=float(trainer_raw.get("weight_decay", 0.0)),
        grad_clip_norm=trainer_raw.get("grad_clip_norm"),
        num_workers=int(trainer_raw.get("num_workers", 0)),
        early_stopping=bool(trainer_raw.get("early_stopping", True)),
        early_stopping_patience=int(trainer_raw.get("early_stopping_patience", 5)),
        monitor=str(trainer_raw.get("monitor", "val_loss")),
        monitor_mode=str(trainer_raw.get("monitor_mode", "min")),
        loss_fn=trainer_raw.get("loss_fn") or None,
    )

    grid_items = []
    for item in hpo_raw.get("grid", []) or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("hpo.grid items must be [key, values]")
        grid_items.append((str(item[0]), list(item[1])))
    hpo = HPOConfig(
        enabled=bool(hpo_raw.get("enabled", False)),
        grid=grid_items,
        metric=str(hpo_raw.get("metric", "val_loss")),
        mode=str(hpo_raw.get("mode", "min")),
    )

    return ExperimentConfig(
        dataset=dataset,
        preprocess=preprocess,
        task=task,
        labels=labels,
        split=split,
        model=model,
        trainer=trainer,
        hpo=hpo,
        output=output,
    )
