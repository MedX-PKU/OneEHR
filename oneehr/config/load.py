from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib

from oneehr.config.schema import (
    AnalysisConfig,
    CalibrationConfig,
    CatBoostConfig,
    DynamicTableConfig,
    DatasetConfig,
    DatasetsConfig,
    EvalBackendConfig,
    EvalConfig,
    EvalSuiteConfig,
    EvalSystemConfig,
    ExperimentConfig,
    GRUConfig,
    HPOConfig,
    LabelTableConfig,
    LabelsConfig,
    LSTMConfig,
    ModelConfig,
    OutputConfig,
    PreprocessConfig,
    StaticTableConfig,
    SplitConfig,
    TaskConfig,
    TCNConfig,
    TrainerConfig,
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
    eval_raw = raw.get("eval", {})
    analysis_raw = raw.get("analysis", {})
    labels_raw = raw.get("labels", {})
    trainer_raw = raw.get("trainer", {})
    hpo_raw = raw.get("hpo", {})
    hpo_models_raw = raw.get("hpo_models", {})
    calibration_raw = raw.get("calibration", {})
    legacy_sections = [name for name in ("cases", "agent") if name in raw]
    if legacy_sections:
        joined = ", ".join(legacy_sections)
        raise ValueError(
            f"Legacy config sections are no longer supported: {joined}. "
            "Use the unified [eval] surface instead."
        )
    if not isinstance(eval_raw, dict):
        raise ValueError("eval must be a table")

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
    )

    import dataclasses as _dc

    _MODEL_CONFIG_MAP: dict[str, type] = {
        "xgboost": XGBoostConfig,
        "catboost": CatBoostConfig,
        "gru": GRUConfig,
        "lstm": LSTMConfig,
        "tcn": TCNConfig,
        "transformer": TransformerConfig,
    }

    def _parse_dataclass(cls: type, raw: dict[str, Any]) -> Any:
        kwargs = {}
        for f in _dc.fields(cls):
            if f.name not in raw:
                continue
            val = raw[f.name]
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
            sub[key] = _parse_dataclass(cls, model_raw_in.get(key, {}))
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
    if model is None and models:
        model = models[0]

    output = OutputConfig(
        root=Path(output_raw.get("root", "logs")),
        run_name=str(output_raw.get("run_name", "run")),
        save_preds=bool(output_raw.get("save_preds", True)),
    )

    def _load_backend_entry(braw: dict[str, Any], *, path_name: str) -> dict[str, Any]:
        headers = braw.get("headers", {}) or {}
        if not isinstance(headers, dict):
            raise ValueError(f"{path_name}.<entry>.headers must be a table")
        return {
            "name": _require(braw, "name"),
            "provider": str(braw.get("provider", "openai_compatible")),
            "base_url": str(braw.get("base_url", "https://api.openai.com/v1")),
            "model": str(_require(braw, "model")),
            "api_key_env": str(braw.get("api_key_env", "OPENAI_API_KEY")),
            "system_prompt": (
                None
                if braw.get("system_prompt") in {None, "", "null"}
                else str(braw.get("system_prompt"))
            ),
            "supports_json_schema": bool(braw.get("supports_json_schema", True)),
            "headers": {str(k): str(v) for k, v in headers.items()},
        }

    def _load_eval_backends(raw_backends: list[dict[str, Any]]) -> list[EvalBackendConfig]:
        backends: list[EvalBackendConfig] = []
        for braw in raw_backends or []:
            if not isinstance(braw, dict):
                raise ValueError("eval.backends entries must be tables")
            entry = _load_backend_entry(braw, path_name="eval.backends")
            entry["prompt_token_cost_per_1k"] = (
                None
                if braw.get("prompt_token_cost_per_1k") in {None, "", "null"}
                else float(braw.get("prompt_token_cost_per_1k"))
            )
            entry["completion_token_cost_per_1k"] = (
                None
                if braw.get("completion_token_cost_per_1k") in {None, "", "null"}
                else float(braw.get("completion_token_cost_per_1k"))
            )
            backends.append(EvalBackendConfig(**entry))
        return backends

    eval_backends_raw = eval_raw.get("backends", []) or []
    eval_systems_raw = eval_raw.get("systems", []) or []
    eval_suites_raw = eval_raw.get("suites", []) or []
    if not isinstance(eval_backends_raw, list):
        raise ValueError("eval.backends must be an array of tables")
    if not isinstance(eval_systems_raw, list):
        raise ValueError("eval.systems must be an array of tables")
    if not isinstance(eval_suites_raw, list):
        raise ValueError("eval.suites must be an array of tables")

    eval_systems: list[EvalSystemConfig] = []
    for sraw in eval_systems_raw:
        if not isinstance(sraw, dict):
            raise ValueError("eval.systems entries must be tables")
        framework_params = sraw.get("framework_params", {}) or {}
        if not isinstance(framework_params, dict):
            raise ValueError("eval.systems.<entry>.framework_params must be a table")
        eval_systems.append(
            EvalSystemConfig(
                name=str(_require(sraw, "name")),
                kind=str(sraw.get("kind", "framework")),
                framework_type=(
                    None
                    if sraw.get("framework_type") in {None, "", "null"}
                    else str(sraw.get("framework_type"))
                ),
                enabled=bool(sraw.get("enabled", True)),
                sample_unit=str(sraw.get("sample_unit", "patient")),
                source_model=(
                    None
                    if sraw.get("source_model") in {None, "", "null"}
                    else str(sraw.get("source_model"))
                ),
                backend_refs=[str(v) for v in sraw.get("backend_refs", []) or []],
                prompt_template=str(sraw.get("prompt_template", "summary_v1")),
                max_samples=(
                    None
                    if sraw.get("max_samples") in {None, "", "null"}
                    else int(sraw.get("max_samples"))
                ),
                max_rounds=int(sraw.get("max_rounds", 1)),
                concurrency=int(sraw.get("concurrency", 1)),
                max_retries=int(sraw.get("max_retries", 2)),
                timeout_seconds=float(sraw.get("timeout_seconds", 60.0)),
                temperature=float(sraw.get("temperature", 0.0)),
                top_p=float(sraw.get("top_p", 1.0)),
                seed=(
                    None
                    if sraw.get("seed") in {None, "", "null"}
                    else int(sraw.get("seed"))
                ),
                framework_params={str(k): v for k, v in framework_params.items()},
            )
        )

    eval_suites: list[EvalSuiteConfig] = []
    for sraw in eval_suites_raw:
        if not isinstance(sraw, dict):
            raise ValueError("eval.suites entries must be tables")
        compare_pairs_raw = sraw.get("compare_pairs", []) or []
        compare_pairs: list[tuple[str, str]] = []
        for item in compare_pairs_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("eval.suites.<entry>.compare_pairs items must be [left, right]")
            compare_pairs.append((str(item[0]), str(item[1])))
        eval_suites.append(
            EvalSuiteConfig(
                name=str(_require(sraw, "name")),
                splits=[str(v) for v in sraw.get("splits", []) or []],
                include_systems=[str(v) for v in sraw.get("include_systems", []) or []],
                primary_metric=(
                    None
                    if sraw.get("primary_metric") in {None, "", "null"}
                    else str(sraw.get("primary_metric"))
                ),
                secondary_metrics=[str(v) for v in sraw.get("secondary_metrics", []) or []],
                slice_by=[str(v) for v in sraw.get("slice_by", []) or []],
                min_coverage=float(sraw.get("min_coverage", 0.0)),
                compare_pairs=compare_pairs,
            )
        )

    eval_cfg = EvalConfig(
        instance_unit=str(eval_raw.get("instance_unit", task.prediction_mode)),
        max_instances=(
            None
            if eval_raw.get("max_instances") in {None, "", "null"}
            else int(eval_raw.get("max_instances"))
        ),
        seed=int(eval_raw.get("seed", 42)),
        include_static=bool(eval_raw.get("include_static", True)),
        include_analysis_context=bool(eval_raw.get("include_analysis_context", False)),
        max_events=int(eval_raw.get("max_events", 200)),
        time_order=str(eval_raw.get("time_order", "asc")),
        slice_by=[str(v) for v in eval_raw.get("slice_by", []) or []],
        primary_metric=(
            None
            if eval_raw.get("primary_metric") in {None, "", "null"}
            else str(eval_raw.get("primary_metric"))
        ),
        bootstrap_samples=int(eval_raw.get("bootstrap_samples", 200)),
        save_evidence=bool(eval_raw.get("save_evidence", True)),
        save_traces=bool(eval_raw.get("save_traces", True)),
        text_render_template=str(eval_raw.get("text_render_template", "summary_v1")),
        backends=_load_eval_backends(eval_backends_raw),
        systems=eval_systems,
        suites=eval_suites,
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
        source=str(calibration_raw.get("source", "val")),
        threshold_strategy=str(calibration_raw.get("threshold_strategy", "f1")),
        use_calibrated=bool(calibration_raw.get("use_calibrated", True)),
    )

    if analysis.top_k <= 0:
        raise ValueError("analysis.top_k must be >= 1")
    if analysis.case_limit <= 0:
        raise ValueError("analysis.case_limit must be >= 1")
    if analysis.shap_max_samples <= 0:
        raise ValueError("analysis.shap_max_samples must be >= 1")
    if "agent_audit" in analysis.default_modules:
        raise ValueError("analysis.default_modules no longer supports 'agent_audit'")
    if eval_cfg.instance_unit not in {"patient", "time"}:
        raise ValueError("eval.instance_unit must be 'patient' or 'time'")
    if eval_cfg.max_instances is not None and eval_cfg.max_instances <= 0:
        raise ValueError("eval.max_instances must be >= 1 when provided")
    if eval_cfg.bootstrap_samples <= 0:
        raise ValueError("eval.bootstrap_samples must be >= 1")
    if eval_cfg.max_events <= 0:
        raise ValueError("eval.max_events must be >= 1")
    if eval_cfg.time_order not in {"asc", "desc"}:
        raise ValueError("eval.time_order must be 'asc' or 'desc'")
    backend_names = [backend.name for backend in eval_cfg.backends]
    if len(set(backend_names)) != len(backend_names):
        raise ValueError("eval.backends names must be unique")
    for backend_cfg in eval_cfg.backends:
        if backend_cfg.provider != "openai_compatible":
            raise ValueError("eval.backends.provider must be 'openai_compatible' in v1")
        if backend_cfg.prompt_token_cost_per_1k is not None and backend_cfg.prompt_token_cost_per_1k < 0:
            raise ValueError("eval.backends.prompt_token_cost_per_1k must be >= 0")
        if backend_cfg.completion_token_cost_per_1k is not None and backend_cfg.completion_token_cost_per_1k < 0:
            raise ValueError("eval.backends.completion_token_cost_per_1k must be >= 0")
    allowed_frameworks = {
        "single_llm",
        "healthcareagent",
        "reconcile",
        "mac",
        "medagent",
        "colacare",
        "mdagents",
    }
    system_names: list[str] = []
    for system_cfg in eval_cfg.systems:
        system_names.append(system_cfg.name)
        if system_cfg.kind not in {"framework", "trained_model"}:
            raise ValueError("eval.systems.kind must be 'framework' or 'trained_model'")
        if system_cfg.sample_unit not in {"patient", "time"}:
            raise ValueError("eval.systems.sample_unit must be 'patient' or 'time'")
        if system_cfg.sample_unit != eval_cfg.instance_unit:
            raise ValueError("eval.systems.sample_unit must match eval.instance_unit")
        if system_cfg.max_samples is not None and system_cfg.max_samples <= 0:
            raise ValueError("eval.systems.max_samples must be >= 1 when provided")
        if system_cfg.max_rounds <= 0:
            raise ValueError("eval.systems.max_rounds must be >= 1")
        if system_cfg.concurrency <= 0:
            raise ValueError("eval.systems.concurrency must be >= 1")
        if system_cfg.kind == "trained_model":
            if system_cfg.source_model in {None, ""}:
                raise ValueError("trained_model systems require source_model")
            if system_cfg.framework_type not in {None, "", "null"}:
                raise ValueError("trained_model systems must not set framework_type")
            if system_cfg.backend_refs:
                raise ValueError("trained_model systems must not set backend_refs")
        else:
            if system_cfg.framework_type not in allowed_frameworks:
                raise ValueError(
                    "framework systems require framework_type in "
                    f"{sorted(allowed_frameworks)!r}"
                )
            unknown_refs = [name for name in system_cfg.backend_refs if name not in set(backend_names)]
            if unknown_refs:
                raise ValueError(f"Unknown eval backend refs: {unknown_refs}")
    if len(set(system_names)) != len(system_names):
        raise ValueError("eval.systems names must be unique")
    for suite_cfg in eval_cfg.suites:
        if suite_cfg.min_coverage < 0.0 or suite_cfg.min_coverage > 1.0:
            raise ValueError("eval.suites.min_coverage must be in [0, 1]")
        unknown_systems = [name for name in suite_cfg.include_systems if name not in set(system_names)]
        if unknown_systems:
            raise ValueError(f"Unknown eval systems in suite {suite_cfg.name!r}: {unknown_systems}")
        for left_name, right_name in suite_cfg.compare_pairs:
            if left_name not in set(system_names) or right_name not in set(system_names):
                raise ValueError(
                    f"Unknown compare_pairs system names in suite {suite_cfg.name!r}: {(left_name, right_name)!r}"
                )

    return ExperimentConfig(
        dataset=dataset,
        datasets=datasets,
        preprocess=preprocess,
        task=task,
        labels=labels,
        split=split,
        model=model,
        models=models,
        trainer=trainer,
        hpo=hpo,
        hpo_by_model=hpo_by_model,
        calibration=calibration,
        eval=eval_cfg,
        analysis=analysis,
        output=output,
    )
