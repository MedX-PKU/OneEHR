from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from oneehr.config.schema import (
    DatasetConfig,
    DatasetsConfig,
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
    datasets_raw = raw.get("datasets")
    preprocess_raw = raw.get("preprocess", {})
    task_raw = _require(raw, "task")
    split_raw = _require(raw, "split")
    model_raw = raw.get("model")
    models_raw = raw.get("models", [])
    if model_raw is None and not models_raw:
        raise ValueError("Missing required key: model or models")
    output_raw = raw.get("output", {})
    labels_raw = raw.get("labels", {})
    trainer_raw = raw.get("trainer", {})
    hpo_raw = raw.get("hpo", {})
    hpo_models_raw = raw.get("hpo_models", {})

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

    datasets = None
    if datasets_raw is not None:
        if not isinstance(datasets_raw, dict):
            raise ValueError("datasets must be a table")
        train_raw = _require(datasets_raw, "train")
        if not isinstance(train_raw, dict):
            raise ValueError("datasets.train must be a table")

        def _load_dataset(ds_raw: dict[str, Any]) -> DatasetConfig:
            return DatasetConfig(
                path=Path(_require(ds_raw, "path")),
                file_type=ds_raw.get("file_type", "csv"),
                patient_id_col=ds_raw.get("patient_id_col", "patient_id"),
                time_col=ds_raw.get("time_col", "event_time"),
                code_col=ds_raw.get("code_col", "code"),
                value_col=ds_raw.get("value_col", "value"),
                label_col=ds_raw.get("label_col", "label"),
                time_format=ds_raw.get("time_format"),
            )

        train_ds = _load_dataset(train_raw)
        test_ds = None
        if datasets_raw.get("test") is not None:
            test_raw = datasets_raw.get("test")
            if not isinstance(test_raw, dict):
                raise ValueError("datasets.test must be a table")
            test_ds = _load_dataset(test_raw)
        datasets = DatasetsConfig(train=train_ds, test=test_ds)

    preprocess = PreprocessConfig(
        bin_size=preprocess_raw.get("bin_size", "1d"),
        numeric_strategy=preprocess_raw.get("numeric_strategy", "mean"),
        categorical_strategy=preprocess_raw.get("categorical_strategy", "count"),
        code_selection=str(preprocess_raw.get("code_selection", "frequency")),
        top_k_codes=(
            None
            if preprocess_raw.get("top_k_codes") in {None, "all"}
            else int(preprocess_raw.get("top_k_codes", 500))
        ),
        min_code_count=int(preprocess_raw.get("min_code_count", 1)),
        code_list=[str(c) for c in preprocess_raw.get("code_list", [])],
        importance_file=(
            None
            if preprocess_raw.get("importance_file") in {None, ""}
            else Path(preprocess_raw.get("importance_file"))
        ),
        importance_code_col=str(preprocess_raw.get("importance_code_col", "code")),
        importance_value_col=str(preprocess_raw.get("importance_value_col", "importance")),
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
        fold_index=split_raw.get("fold_index"),
    )

    def _load_model(model_raw_in: dict[str, Any]) -> ModelConfig:
        xgb_raw = model_raw_in.get("xgboost", {})
        gru_raw = model_raw_in.get("gru", {})
        rnn_raw = model_raw_in.get("rnn", {})
        tf_raw = model_raw_in.get("transformer", {})
        return ModelConfig(
            name=_require(model_raw_in, "name"),
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

    model = None
    if model_raw is not None:
        model = _load_model(model_raw)
    models = []
    for mraw in models_raw or []:
        if not isinstance(mraw, dict):
            raise ValueError("models entries must be tables")
        models.append(_load_model(mraw))
    if model is not None and not models:
        models = [model]

    output = OutputConfig(
        root=Path(output_raw.get("root", "logs")),
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

    def _load_hpo(hpo_section: dict[str, Any]) -> HPOConfig:
        grid_items = []
        for item in hpo_section.get("grid", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("hpo.grid items must be [key, values]")
            grid_items.append((str(item[0]), list(item[1])))
        return HPOConfig(
            enabled=bool(hpo_section.get("enabled", False)),
            grid=grid_items,
            metric=str(hpo_section.get("metric", "val_loss")),
            mode=str(hpo_section.get("mode", "min")),
        )

    hpo = _load_hpo(hpo_raw)
    hpo_by_model: dict[str, HPOConfig] = {}
    for name, section in (hpo_models_raw or {}).items():
        if not isinstance(section, dict):
            raise ValueError("hpo_models entries must be tables")
        hpo_by_model[str(name)] = _load_hpo(section)

    return ExperimentConfig(
        dataset=dataset,
        datasets=datasets,
        preprocess=preprocess,
        task=task,
        labels=labels,
        split=split,
        model=model if model is not None else models[0],
        models=models,
        trainer=trainer,
        hpo=hpo,
        hpo_by_model=hpo_by_model,
        output=output,
    )
