from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from oneehr.config.schema import (
    AgentBackendConfig,
    AgentConfig,
    AgentPredictConfig,
    AgentPredictOutputConfig,
    AgentPredictPromptConfig,
    AgentReviewConfig,
    AgentReviewPromptConfig,
    AnalysisConfig,
    CalibrationConfig,
    CasesConfig,
    DynamicTableConfig,
    DatasetConfig,
    DatasetsConfig,
    ExperimentConfig,
    HPOConfig,
    LabelTableConfig,
    LabelsConfig,
    ModelConfig,
    OutputConfig,
    PreprocessConfig,
    StaticTableConfig,
    SplitConfig,
    TaskConfig,
    TrainerConfig,
    GRUConfig,
    LSTMConfig,
    RNNConfig,
    TransformerConfig,
    XGBoostConfig,
    AdaCareConfig,
    DrAgentConfig,
    CatBoostConfig,
    ConCareConfig,
    DTConfig,
    GBDTConfig,
    GRASPConfig,
    MCGRUConfig,
    MLPConfig,
    RETAINConfig,
    RFConfig,
    StageNetConfig,
    TCNConfig,
)


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    return d[key]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    dataset_raw = raw.get("dataset")
    datasets_raw = raw.get("datasets")
    if dataset_raw is None and datasets_raw is None:
        raise ValueError("Missing required key: dataset or datasets")
    preprocess_raw = raw.get("preprocess", {})
    task_raw = _require(raw, "task")
    split_raw = _require(raw, "split")
    model_raw = raw.get("model")
    models_raw = raw.get("models", [])
    output_raw = raw.get("output", {})
    analysis_raw = raw.get("analysis", {})
    cases_raw = raw.get("cases", {})
    agent_raw = raw.get("agent", {})
    labels_raw = raw.get("labels", {})
    trainer_raw = raw.get("trainer", {})
    hpo_raw = raw.get("hpo", {})
    hpo_models_raw = raw.get("hpo_models", {})
    calibration_raw = raw.get("calibration", {})
    if not isinstance(cases_raw, dict):
        raise ValueError("cases must be a table")
    if not isinstance(agent_raw, dict):
        raise ValueError("agent must be a table")
    agent_predict_raw = agent_raw.get("predict", {}) or {}
    agent_review_raw = agent_raw.get("review", {}) or {}
    if not isinstance(agent_predict_raw, dict):
        raise ValueError("agent.predict must be a table")
    if not isinstance(agent_review_raw, dict):
        raise ValueError("agent.review must be a table")
    agent_predict_enabled_raw = bool(agent_predict_raw.get("enabled", False))
    if model_raw is None and not models_raw and not agent_predict_enabled_raw:
        raise ValueError("Missing required key: model or models")

    dataset = None
    if dataset_raw is not None:
        if not isinstance(dataset_raw, dict):
            raise ValueError("dataset must be a table")
        dynamic = None
        if dataset_raw.get("dynamic") not in {None, ""}:
            dynamic = DynamicTableConfig(path=Path(dataset_raw.get("dynamic")))

        static = None
        if dataset_raw.get("static") not in {None, ""}:
            static = StaticTableConfig(path=Path(dataset_raw.get("static")))
        if dynamic is None and static is None:
            raise ValueError("dataset.dynamic or dataset.static is required.")

        label = None
        if dataset_raw.get("label") not in {None, ""}:
            label = LabelTableConfig(path=Path(dataset_raw.get("label")))

        dataset = DatasetConfig(dynamic=dynamic, static=static, label=label)

    datasets = None
    if datasets_raw is not None:
        if not isinstance(datasets_raw, dict):
            raise ValueError("datasets must be a table")
        train_raw = _require(datasets_raw, "train")
        if not isinstance(train_raw, dict):
            raise ValueError("datasets.train must be a table")

        def _load_dataset(ds_raw: dict[str, Any]) -> DatasetConfig:
            dynamic = None
            if ds_raw.get("dynamic") not in {None, ""}:
                dynamic = DynamicTableConfig(path=Path(ds_raw.get("dynamic")))

            static = None
            if ds_raw.get("static") not in {None, ""}:
                static = StaticTableConfig(path=Path(ds_raw.get("static")))
            if dynamic is None and static is None:
                raise ValueError("datasets.<split>.dynamic or datasets.<split>.static is required.")

            label = None
            if ds_raw.get("label") not in {None, ""}:
                label = LabelTableConfig(path=Path(ds_raw.get("label")))

            return DatasetConfig(dynamic=dynamic, static=static, label=label)

        train_ds = _load_dataset(train_raw)
        test_ds = None
        if datasets_raw.get("test") is not None:
            test_raw = datasets_raw.get("test")
            if not isinstance(test_raw, dict):
                raise ValueError("datasets.test must be a table")
            test_ds = _load_dataset(test_raw)
        datasets = DatasetsConfig(train=train_ds, test=test_ds)
        if dataset is None:
            dataset = train_ds

    preprocess = PreprocessConfig(
        bin_size=preprocess_raw.get("bin_size", "1d"),
        numeric_strategy=preprocess_raw.get("numeric_strategy", "mean"),
        categorical_strategy=preprocess_raw.get("categorical_strategy", "onehot"),
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
        pipeline=list(preprocess_raw.get("pipeline", []) or []),
    )

    task = TaskConfig(
        kind=_require(task_raw, "kind"),
        prediction_mode=task_raw.get("prediction_mode", "patient"),
    )

    labels = LabelsConfig(
        fn=labels_raw.get("fn") or None,
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
        inner_kind=split_raw.get("inner_kind"),
        inner_n_splits=split_raw.get("inner_n_splits"),
    )

    import dataclasses as _dc

    # Fields whose ``None`` default needs special TOML handling (empty string / "null").
    _NULLABLE_INT_FIELDS = frozenset({
        ("rf", "max_depth"),
        ("dt", "max_depth"),
    })

    _MODEL_CONFIG_MAP: dict[str, type] = {
        "xgboost": XGBoostConfig,
        "catboost": CatBoostConfig,
        "rf": RFConfig,
        "dt": DTConfig,
        "gbdt": GBDTConfig,
        "gru": GRUConfig,
        "rnn": RNNConfig,
        "lstm": LSTMConfig,
        "mlp": MLPConfig,
        "tcn": TCNConfig,
        "transformer": TransformerConfig,
        "adacare": AdaCareConfig,
        "stagenet": StageNetConfig,
        "retain": RETAINConfig,
        "concare": ConCareConfig,
        "grasp": GRASPConfig,
        "mcgru": MCGRUConfig,
        "dragent": DrAgentConfig,
    }

    def _parse_dataclass(section_name: str, cls: type, raw: dict[str, Any]) -> Any:
        kwargs = {}
        for f in _dc.fields(cls):
            if f.name not in raw:
                continue
            val = raw[f.name]
            # Special handling for nullable int fields.
            if (section_name, f.name) in _NULLABLE_INT_FIELDS:
                if val in {None, "", "null"}:
                    kwargs[f.name] = None
                    continue
                kwargs[f.name] = int(val)
                continue
            # Coerce to the field's declared type.
            if f.type in ("int", int):
                kwargs[f.name] = int(val)
            elif f.type in ("float", float):
                kwargs[f.name] = float(val)
            elif f.type in ("bool", bool):
                kwargs[f.name] = bool(val)
            elif f.type in ("str", str):
                kwargs[f.name] = str(val)
            else:
                kwargs[f.name] = val
        return cls(**kwargs)

    def _load_model(model_raw_in: dict[str, Any]) -> ModelConfig:
        sub = {}
        for key, cls in _MODEL_CONFIG_MAP.items():
            sub[key] = _parse_dataclass(key, cls, model_raw_in.get(key, {}))
        return ModelConfig(name=_require(model_raw_in, "name"), **sub)

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
    if model is None and not models and agent_predict_enabled_raw:
        model = ModelConfig(name="_agent_placeholder")
    elif model is None and models:
        model = models[0]

    output = OutputConfig(
        root=Path(output_raw.get("root", "logs")),
        run_name=str(output_raw.get("run_name", "run")),
        save_preds=bool(output_raw.get("save_preds", True)),
    )

    if not isinstance(analysis_raw, dict):
        raise ValueError("analysis must be a table")
    analysis = AnalysisConfig(
        default_modules=[str(m) for m in analysis_raw.get("default_modules", [])] or AnalysisConfig().default_modules,
        top_k=int(analysis_raw.get("top_k", 20)),
        stratify_by=[str(c) for c in analysis_raw.get("stratify_by", [])],
        case_limit=int(analysis_raw.get("case_limit", 50)),
        save_plot_specs=bool(analysis_raw.get("save_plot_specs", True)),
        shap_max_samples=int(analysis_raw.get("shap_max_samples", 500)),
    )

    cases = CasesConfig(
        include_static=bool(cases_raw.get("include_static", True)),
        include_analysis_refs=bool(cases_raw.get("include_analysis_refs", True)),
        history_window=(
            None
            if cases_raw.get("history_window") in {None, "", "null"}
            else str(cases_raw.get("history_window"))
        ),
        max_events=int(cases_raw.get("max_events", 200)),
        time_order=str(cases_raw.get("time_order", "asc")),
        case_limit=(
            None
            if cases_raw.get("case_limit") in {None, "", "null"}
            else int(cases_raw.get("case_limit"))
        ),
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
        final_refit=str(trainer_raw.get("final_refit", "train_val")),
        final_model_source=str(trainer_raw.get("final_model_source", "refit")),
        bootstrap_test=bool(trainer_raw.get("bootstrap_test", False)),
        bootstrap_n=int(trainer_raw.get("bootstrap_n", 200)),
        repeat=int(trainer_raw.get("repeat", 1)),
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
            scope=str(hpo_section.get("scope", "single")),
            tune_split=hpo_section.get("tune_split"),
            aggregate_metric=hpo_section.get("aggregate_metric"),
        )

    hpo = _load_hpo(hpo_raw)
    hpo_by_model: dict[str, HPOConfig] = {}
    for name, section in (hpo_models_raw or {}).items():
        if not isinstance(section, dict):
            raise ValueError("hpo_models entries must be tables")
        hpo_by_model[str(name)] = _load_hpo(section)

    calibration = CalibrationConfig(
        enabled=bool(calibration_raw.get("enabled", False)),
        method=str(calibration_raw.get("method", "temperature")),
        source=str(calibration_raw.get("source", "val")),
        threshold_strategy=str(calibration_raw.get("threshold_strategy", "f1")),
        use_calibrated=bool(calibration_raw.get("use_calibrated", True)),
    )

    def _load_agent_backends(raw_backends: list[dict[str, Any]], *, path_name: str) -> list[AgentBackendConfig]:
        backends: list[AgentBackendConfig] = []
        for braw in raw_backends or []:
            if not isinstance(braw, dict):
                raise ValueError(f"{path_name} entries must be tables")
            headers = braw.get("headers", {}) or {}
            if not isinstance(headers, dict):
                raise ValueError(f"{path_name}.<entry>.headers must be a table")
            backends.append(
                AgentBackendConfig(
                    name=_require(braw, "name"),
                    provider=str(braw.get("provider", "openai_compatible")),
                    base_url=str(braw.get("base_url", "https://api.openai.com/v1")),
                    model=str(_require(braw, "model")),
                    api_key_env=str(braw.get("api_key_env", "OPENAI_API_KEY")),
                    system_prompt=(
                        None
                        if braw.get("system_prompt") in {None, "", "null"}
                        else str(braw.get("system_prompt"))
                    ),
                    supports_json_schema=bool(braw.get("supports_json_schema", True)),
                    headers={str(k): str(v) for k, v in headers.items()},
                )
            )
        return backends

    agent_predict_prompt_raw = agent_predict_raw.get("prompt", {}) or {}
    agent_predict_output_raw = agent_predict_raw.get("output", {}) or {}
    agent_predict_backends_raw = agent_predict_raw.get("backends", []) or []
    if not isinstance(agent_predict_prompt_raw, dict):
        raise ValueError("agent.predict.prompt must be a table")
    if not isinstance(agent_predict_output_raw, dict):
        raise ValueError("agent.predict.output must be a table")
    if not isinstance(agent_predict_backends_raw, list):
        raise ValueError("agent.predict.backends must be an array of tables")
    agent_predict_prompt = AgentPredictPromptConfig(
        include_static=bool(agent_predict_prompt_raw.get("include_static", True)),
        include_labels_context=bool(agent_predict_prompt_raw.get("include_labels_context", False)),
        history_window=(
            None
            if agent_predict_prompt_raw.get("history_window") in {None, "", "null"}
            else str(agent_predict_prompt_raw.get("history_window"))
        ),
        max_events=int(agent_predict_prompt_raw.get("max_events", 200)),
        time_order=str(agent_predict_prompt_raw.get("time_order", "asc")),
        sections=[str(s) for s in agent_predict_prompt_raw.get("sections", [])] or AgentPredictPromptConfig().sections,
    )
    agent_predict_output = AgentPredictOutputConfig(
        include_explanation=bool(agent_predict_output_raw.get("include_explanation", True)),
        include_confidence=bool(agent_predict_output_raw.get("include_confidence", False)),
    )
    agent_predict = AgentPredictConfig(
        enabled=bool(agent_predict_raw.get("enabled", False)),
        sample_unit=str(agent_predict_raw.get("sample_unit", "patient")),
        prompt_template=str(agent_predict_raw.get("prompt_template", "summary_v1")),
        json_schema_version=int(agent_predict_raw.get("json_schema_version", 1)),
        max_samples=(
            None
            if agent_predict_raw.get("max_samples") in {None, "", "null"}
            else int(agent_predict_raw.get("max_samples"))
        ),
        save_prompts=bool(agent_predict_raw.get("save_prompts", True)),
        save_responses=bool(agent_predict_raw.get("save_responses", True)),
        save_parsed=bool(agent_predict_raw.get("save_parsed", True)),
        concurrency=int(agent_predict_raw.get("concurrency", 1)),
        max_retries=int(agent_predict_raw.get("max_retries", 2)),
        timeout_seconds=float(agent_predict_raw.get("timeout_seconds", 60.0)),
        temperature=float(agent_predict_raw.get("temperature", 0.0)),
        top_p=float(agent_predict_raw.get("top_p", 1.0)),
        seed=(
            None
            if agent_predict_raw.get("seed") in {None, "", "null"}
            else int(agent_predict_raw.get("seed"))
        ),
        prompt=agent_predict_prompt,
        output=agent_predict_output,
        backends=_load_agent_backends(agent_predict_backends_raw, path_name="agent.predict.backends"),
    )

    agent_review_prompt_raw = agent_review_raw.get("prompt", {}) or {}
    agent_review_backends_raw = agent_review_raw.get("backends", []) or []
    if not isinstance(agent_review_prompt_raw, dict):
        raise ValueError("agent.review.prompt must be a table")
    if not isinstance(agent_review_backends_raw, list):
        raise ValueError("agent.review.backends must be an array of tables")
    agent_review_prompt = AgentReviewPromptConfig(
        include_static=bool(agent_review_prompt_raw.get("include_static", True)),
        include_ground_truth=bool(agent_review_prompt_raw.get("include_ground_truth", True)),
        include_analysis_context=bool(agent_review_prompt_raw.get("include_analysis_context", True)),
        max_events=int(agent_review_prompt_raw.get("max_events", 100)),
        time_order=str(agent_review_prompt_raw.get("time_order", "asc")),
        sections=[str(s) for s in agent_review_prompt_raw.get("sections", [])] or AgentReviewPromptConfig().sections,
    )
    agent_review = AgentReviewConfig(
        enabled=bool(agent_review_raw.get("enabled", False)),
        prompt_template=str(agent_review_raw.get("prompt_template", "evidence_review_v1")),
        json_schema_version=int(agent_review_raw.get("json_schema_version", 1)),
        prediction_origins=[
            str(s) for s in agent_review_raw.get("prediction_origins", [])
        ] or AgentReviewConfig().prediction_origins,
        max_cases=(
            None
            if agent_review_raw.get("max_cases") in {None, "", "null"}
            else int(agent_review_raw.get("max_cases"))
        ),
        save_prompts=bool(agent_review_raw.get("save_prompts", True)),
        save_responses=bool(agent_review_raw.get("save_responses", True)),
        save_parsed=bool(agent_review_raw.get("save_parsed", True)),
        concurrency=int(agent_review_raw.get("concurrency", 1)),
        max_retries=int(agent_review_raw.get("max_retries", 2)),
        timeout_seconds=float(agent_review_raw.get("timeout_seconds", 60.0)),
        temperature=float(agent_review_raw.get("temperature", 0.0)),
        top_p=float(agent_review_raw.get("top_p", 1.0)),
        seed=(
            None
            if agent_review_raw.get("seed") in {None, "", "null"}
            else int(agent_review_raw.get("seed"))
        ),
        prompt=agent_review_prompt,
        backends=_load_agent_backends(agent_review_backends_raw, path_name="agent.review.backends"),
    )
    agent = AgentConfig(predict=agent_predict, review=agent_review)

    if agent.predict.enabled:
        if agent.predict.sample_unit not in {"patient", "time"}:
            raise ValueError("agent.predict.sample_unit must be 'patient' or 'time'")
        if agent.predict.prompt.time_order not in {"asc", "desc"}:
            raise ValueError("agent.predict.prompt.time_order must be 'asc' or 'desc'")
        if not agent.predict.backends:
            raise ValueError("agent.predict.enabled=true requires at least one [[agent.predict.backends]] entry")
    for backend_cfg in agent.predict.backends:
        if backend_cfg.provider != "openai_compatible":
            raise ValueError("agent.predict.backends.provider must be 'openai_compatible' in v1")
    if agent.review.enabled:
        if agent.review.prompt.time_order not in {"asc", "desc"}:
            raise ValueError("agent.review.prompt.time_order must be 'asc' or 'desc'")
        if not agent.review.backends:
            raise ValueError("agent.review.enabled=true requires at least one [[agent.review.backends]] entry")
        invalid_origins = [origin for origin in agent.review.prediction_origins if origin not in {"model", "agent"}]
        if invalid_origins:
            raise ValueError("agent.review.prediction_origins must contain only 'model' and/or 'agent'")
    for backend_cfg in agent.review.backends:
        if backend_cfg.provider != "openai_compatible":
            raise ValueError("agent.review.backends.provider must be 'openai_compatible' in v1")
    if analysis.top_k <= 0:
        raise ValueError("analysis.top_k must be >= 1")
    if analysis.case_limit <= 0:
        raise ValueError("analysis.case_limit must be >= 1")
    if analysis.shap_max_samples <= 0:
        raise ValueError("analysis.shap_max_samples must be >= 1")
    if cases.max_events <= 0:
        raise ValueError("cases.max_events must be >= 1")
    if cases.time_order not in {"asc", "desc"}:
        raise ValueError("cases.time_order must be 'asc' or 'desc'")
    if cases.case_limit is not None and cases.case_limit <= 0:
        raise ValueError("cases.case_limit must be >= 1 when provided")
    if agent.review.prompt.max_events <= 0:
        raise ValueError("agent.review.prompt.max_events must be >= 1")
    if agent.review.max_cases is not None and agent.review.max_cases <= 0:
        raise ValueError("agent.review.max_cases must be >= 1 when provided")

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
        calibration=calibration,
        analysis=analysis,
        cases=cases,
        agent=agent,
        output=output,
    )
