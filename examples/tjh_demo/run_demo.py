from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_TRAIN_RAW = Path("/home/yhzhu/projects/pyehr/datasets/tjh/raw/time_series_375_prerpocess_en.xlsx")
DEFAULT_EXTERNAL_RAW = Path("/home/yhzhu/projects/pyehr/datasets/tjh/raw/time_series_test_110_preprocess_en.xlsx")
DEFAULT_BASE_URL = "https://zenmux.ai/api/v1"
DEFAULT_REQUESTED_MODEL = "openai/gpt-5.4"
DEFAULT_FALLBACK_MODEL = "openai/gpt-5"

FAST_MODULES = [
    "dataset_profile",
    "cohort_analysis",
    "prediction_audit",
    "test_audit",
]
SHOWCASE_MODULES = [
    "dataset_profile",
    "cohort_analysis",
    "prediction_audit",
    "test_audit",
    "temporal_analysis",
    "interpretability",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-TJH OneEHR tutorial presets end-to-end.")
    parser.add_argument("--preset", choices=["fast", "showcase"], required=True, help="Preset to execute.")
    parser.add_argument("--train-raw", type=Path, default=DEFAULT_TRAIN_RAW, help="Path to the TJH training Excel file.")
    parser.add_argument(
        "--external-raw",
        type=Path,
        default=DEFAULT_EXTERNAL_RAW,
        help="Path to the TJH external-test Excel file.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".cache/tjh_demo"),
        help="Cache root for standardized data, configs, and manifests.",
    )
    parser.add_argument("--logs-root", type=Path, default=Path("logs"), help="Root directory for OneEHR run outputs.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip Excel-to-CSV conversion if cache already exists.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing training, test, analysis, and eval outputs where supported.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Zenmux OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--requested-model",
        default=DEFAULT_REQUESTED_MODEL,
        help="Requested Zenmux model slug. The script falls back to openai/gpt-5 on unsupported-model errors.",
    )
    parser.add_argument(
        "--fallback-model",
        default=DEFAULT_FALLBACK_MODEL,
        help="Fallback Zenmux model slug used only if the requested model is rejected by the provider.",
    )
    parser.add_argument(
        "--capture-screenshots",
        action="store_true",
        help="After a successful showcase run, invoke capture_screenshots.sh.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Web UI port used when capturing screenshots.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_root = (repo_root / args.cache_root).resolve() if not args.cache_root.is_absolute() else args.cache_root.resolve()
    logs_root = (repo_root / args.logs_root).resolve() if not args.logs_root.is_absolute() else args.logs_root.resolve()

    if not args.skip_prepare:
        _run_prepare(
            repo_root=repo_root,
            train_raw=args.train_raw.resolve(),
            external_raw=args.external_raw.resolve(),
            cache_root=cache_root,
        )

    if args.preset == "showcase":
        fast_run_root = logs_root / "tjh_tutorial_fast"
        if not fast_run_root.exists():
            print("Fast baseline is missing. Building it before the showcase preset.", flush=True)
            _run_preset(
                repo_root=repo_root,
                preset="fast",
                cache_root=cache_root,
                logs_root=logs_root,
                force=args.force,
                resolved_model=None,
                fallback_note=None,
            )

    resolved_model = None
    fallback_note = None
    if args.preset == "showcase":
        resolved_model, fallback_note = _resolve_zenmux_model(
            base_url=args.base_url,
            api_key_env="ZENMUX_API_KEY",
            requested_model=args.requested_model,
            fallback_model=args.fallback_model,
        )
        print(f"Resolved Zenmux model: {resolved_model}", flush=True)
        if fallback_note is not None:
            print(json.dumps(fallback_note, indent=2), flush=True)

    run_root = _run_preset(
        repo_root=repo_root,
        preset=args.preset,
        cache_root=cache_root,
        logs_root=logs_root,
        force=args.force,
        resolved_model=resolved_model,
        fallback_note=fallback_note,
        base_url=args.base_url,
    )

    result = {
        "status": "ok",
        "preset": args.preset,
        "run_root": str(run_root),
        "config_path": str(cache_root / "configs" / f"{args.preset}.toml"),
        "resolved_model": resolved_model,
    }
    print(json.dumps(result, indent=2))

    if args.capture_screenshots:
        if args.preset != "showcase":
            raise SystemExit("--capture-screenshots is only supported for the showcase preset.")
        script_path = repo_root / "examples" / "tjh_demo" / "capture_screenshots.sh"
        subprocess.check_call(
            [
                str(script_path),
                "--run-name",
                "tjh_tutorial_showcase",
                "--baseline-run",
                "tjh_tutorial_fast",
                "--port",
                str(args.port),
            ],
            cwd=repo_root,
        )


def _run_preset(
    *,
    repo_root: Path,
    preset: str,
    cache_root: Path,
    logs_root: Path,
    force: bool,
    resolved_model: str | None,
    fallback_note: dict[str, Any] | None,
    base_url: str = DEFAULT_BASE_URL,
) -> Path:
    config_path = cache_root / "configs" / f"{preset}.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    standardized_root = cache_root / "standardized"
    train_dir = standardized_root / "train"
    external_dir = standardized_root / "external"
    if not train_dir.exists() or not external_dir.exists():
        raise FileNotFoundError(
            "Standardized TJH CSVs are missing. Run prepare_data.py first or omit --skip-prepare."
        )

    if preset == "fast":
        run_name = "tjh_tutorial_fast"
        config_text = _build_fast_config(
            train_dir=train_dir,
            external_dir=external_dir,
            logs_root=logs_root,
            run_name=run_name,
        )
        config_path.write_text(config_text, encoding="utf-8")
        _run_cli(repo_root, "preprocess", "--config", str(config_path), "--overview")
        _run_cli(repo_root, "train", "--config", str(config_path), *(_force_flag(force)))
        _run_cli(repo_root, "test", "--config", str(config_path), *(_force_flag(force)))
        _run_cli(
            repo_root,
            "analyze",
            "--config",
            str(config_path),
            *sum((["--module", name] for name in FAST_MODULES), []),
        )
        _write_run_manifest_note(
            cache_root=cache_root,
            preset=preset,
            run_name=run_name,
            config_path=config_path,
            resolved_model=None,
            fallback_note=None,
        )
        return logs_root / run_name

    if resolved_model is None:
        raise ValueError("showcase preset requires a resolved Zenmux model")

    run_name = "tjh_tutorial_showcase"
    config_text = _build_showcase_config(
        train_dir=train_dir,
        external_dir=external_dir,
        logs_root=logs_root,
        run_name=run_name,
        base_url=base_url,
        resolved_model=resolved_model,
    )
    config_path.write_text(config_text, encoding="utf-8")
    _run_cli(repo_root, "preprocess", "--config", str(config_path), "--overview")
    _run_cli(repo_root, "train", "--config", str(config_path), *(_force_flag(force)))
    _run_cli(repo_root, "test", "--config", str(config_path), *(_force_flag(force)))
    _run_cli(
        repo_root,
        "analyze",
        "--config",
        str(config_path),
        *sum((["--module", name] for name in SHOWCASE_MODULES), []),
        "--method",
        "xgboost",
    )
    _run_cli(repo_root, "eval", "build", "--config", str(config_path), *(_force_flag(force)))
    _run_cli(repo_root, "eval", "run", "--config", str(config_path), *(_force_flag(force)))
    _run_cli(repo_root, "eval", "report", "--config", str(config_path), *(_force_flag(force)))
    _run_cli(
        repo_root,
        "analyze",
        "--config",
        str(config_path),
        *sum((["--module", name] for name in SHOWCASE_MODULES), []),
        "--compare-run",
        str(logs_root / "tjh_tutorial_fast"),
        "--method",
        "xgboost",
    )
    _write_run_manifest_note(
        cache_root=cache_root,
        preset=preset,
        run_name=run_name,
        config_path=config_path,
        resolved_model=resolved_model,
        fallback_note=fallback_note,
    )
    return logs_root / run_name


def _run_prepare(*, repo_root: Path, train_raw: Path, external_raw: Path, cache_root: Path) -> None:
    script_path = repo_root / "examples" / "tjh_demo" / "prepare_data.py"
    subprocess.check_call(
        [
            sys.executable,
            str(script_path),
            "--train-raw",
            str(train_raw),
            "--external-raw",
            str(external_raw),
            "--cache-root",
            str(cache_root),
        ],
        cwd=repo_root,
    )


def _run_cli(repo_root: Path, *args: str) -> None:
    cli_path = Path(sys.executable).with_name("oneehr")
    if not cli_path.exists():
        resolved = shutil.which("oneehr")
        if resolved is None:
            raise FileNotFoundError("Could not find the `oneehr` CLI executable in the current environment.")
        cli_path = Path(resolved)
    print("$", str(cli_path), *args, flush=True)
    subprocess.check_call([str(cli_path), *args], cwd=repo_root)


def _force_flag(force: bool) -> tuple[str, ...]:
    return ("--force",) if force else ()


def _build_fast_config(*, train_dir: Path, external_dir: Path, logs_root: Path, run_name: str) -> str:
    return textwrap.dedent(
        f"""
        [datasets.train]
        dynamic = "{train_dir / 'dynamic.csv'}"
        static = "{train_dir / 'static.csv'}"
        label = "{train_dir / 'label.csv'}"

        [datasets.test]
        dynamic = "{external_dir / 'dynamic.csv'}"
        static = "{external_dir / 'static.csv'}"
        label = "{external_dir / 'label.csv'}"

        [preprocess]
        bin_size = "1d"
        numeric_strategy = "mean"
        categorical_strategy = "onehot"
        code_selection = "frequency"
        top_k_codes = 80
        min_code_count = 1
        pipeline = [
          {{ op = "standardize", cols = "num__*" }},
          {{ op = "impute", strategy = "mean", cols = "num__*" }},
        ]

        [task]
        kind = "binary"
        prediction_mode = "patient"

        [labels]
        fn = "examples/tjh_pipeline/label_fn.py:build_labels"
        bin_from_time_col = true

        [split]
        kind = "random"
        seed = 42
        val_size = 0.15
        test_size = 0.2

        [[models]]
        name = "xgboost"

        [[models]]
        name = "rf"

        [trainer]
        device = "cpu"
        seed = 42
        max_epochs = 1
        batch_size = 32
        lr = 1e-3
        weight_decay = 0.0
        early_stopping = true
        early_stopping_patience = 1
        monitor = "val_loss"
        monitor_mode = "min"
        final_refit = "train_val"
        final_model_source = "refit"
        bootstrap_test = false
        bootstrap_n = 20

        [analysis]
        default_modules = ["dataset_profile", "cohort_analysis", "prediction_audit", "test_audit"]
        top_k = 15
        stratify_by = ["Sex"]
        case_limit = 12
        save_plot_specs = true
        shap_max_samples = 64

        [output]
        root = "{logs_root}"
        run_name = "{run_name}"
        save_preds = true
        """
    ).strip() + "\n"


def _build_showcase_config(
    *,
    train_dir: Path,
    external_dir: Path,
    logs_root: Path,
    run_name: str,
    base_url: str,
    resolved_model: str,
) -> str:
    return textwrap.dedent(
        f"""
        [datasets.train]
        dynamic = "{train_dir / 'dynamic.csv'}"
        static = "{train_dir / 'static.csv'}"
        label = "{train_dir / 'label.csv'}"

        [datasets.test]
        dynamic = "{external_dir / 'dynamic.csv'}"
        static = "{external_dir / 'static.csv'}"
        label = "{external_dir / 'label.csv'}"

        [preprocess]
        bin_size = "1d"
        numeric_strategy = "mean"
        categorical_strategy = "onehot"
        code_selection = "frequency"
        top_k_codes = 200
        min_code_count = 1
        pipeline = [
          {{ op = "standardize", cols = "num__*" }},
          {{ op = "impute", strategy = "mean", cols = "num__*" }},
        ]

        [task]
        kind = "binary"
        prediction_mode = "patient"

        [labels]
        fn = "examples/tjh_pipeline/label_fn.py:build_labels"
        bin_from_time_col = true

        [split]
        kind = "random"
        seed = 42
        val_size = 0.15
        test_size = 0.2

        [[models]]
        name = "xgboost"

        [[models]]
        name = "rf"

        [trainer]
        device = "cpu"
        seed = 42
        max_epochs = 4
        batch_size = 32
        lr = 1e-3
        weight_decay = 0.0
        early_stopping = true
        early_stopping_patience = 2
        monitor = "val_loss"
        monitor_mode = "min"
        final_refit = "train_val"
        final_model_source = "refit"
        bootstrap_test = false
        bootstrap_n = 50

        [calibration]
        enabled = true
        method = "temperature"
        source = "val"
        threshold_strategy = "f1"
        use_calibrated = true

        [analysis]
        default_modules = ["dataset_profile", "cohort_analysis", "prediction_audit", "test_audit", "temporal_analysis", "interpretability"]
        top_k = 20
        stratify_by = ["Sex"]
        case_limit = 18
        save_plot_specs = true
        shap_max_samples = 128

        [eval]
        instance_unit = "patient"
        max_instances = 18
        seed = 42
        include_static = true
        include_analysis_context = true
        max_events = 120
        time_order = "desc"
        primary_metric = "accuracy"
        bootstrap_samples = 32
        save_evidence = true
        save_traces = true
        text_render_template = "summary_v1"

        [[eval.backends]]
        name = "zenmux"
        provider = "openai_compatible"
        base_url = "{base_url}"
        model = "{resolved_model}"
        api_key_env = "ZENMUX_API_KEY"
        supports_json_schema = false

        [[eval.systems]]
        name = "xgboost_ref"
        kind = "trained_model"
        sample_unit = "patient"
        source_model = "xgboost"

        [[eval.systems]]
        name = "single_llm_eval"
        kind = "framework"
        framework_type = "single_llm"
        sample_unit = "patient"
        backend_refs = ["zenmux"]
        max_samples = 18
        concurrency = 1
        max_retries = 2
        timeout_seconds = 90.0
        temperature = 0.0
        top_p = 1.0

        [[eval.systems]]
        name = "reconcile_eval"
        kind = "framework"
        framework_type = "reconcile"
        sample_unit = "patient"
        backend_refs = ["zenmux"]
        max_samples = 18
        max_rounds = 2
        concurrency = 1
        max_retries = 2
        timeout_seconds = 90.0
        temperature = 0.0
        top_p = 1.0

        [[eval.systems]]
        name = "mdagents_eval"
        kind = "framework"
        framework_type = "mdagents"
        sample_unit = "patient"
        backend_refs = ["zenmux"]
        max_samples = 18
        max_rounds = 2
        concurrency = 1
        max_retries = 2
        timeout_seconds = 90.0
        temperature = 0.0
        top_p = 1.0
        framework_params = {{ num_teams_advanced = 2, num_agents_per_team_advanced = 2 }}

        [[eval.suites]]
        name = "tutorial_core"
        primary_metric = "accuracy"
        include_systems = ["xgboost_ref", "single_llm_eval", "reconcile_eval", "mdagents_eval"]
        compare_pairs = [
          ["xgboost_ref", "single_llm_eval"],
          ["xgboost_ref", "reconcile_eval"],
          ["xgboost_ref", "mdagents_eval"],
        ]

        [output]
        root = "{logs_root}"
        run_name = "{run_name}"
        save_preds = true
        """
    ).strip() + "\n"


def _resolve_zenmux_model(
    *,
    base_url: str,
    api_key_env: str,
    requested_model: str,
    fallback_model: str,
) -> tuple[str, dict[str, Any] | None]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing required environment variable: {api_key_env}")

    catalog = _fetch_model_catalog(base_url=base_url, api_key=api_key)
    try:
        _probe_chat_completion(base_url=base_url, api_key=api_key, model=requested_model)
        if requested_model in catalog:
            return requested_model, None
        return requested_model, {"requested_model_missing_from_catalog": requested_model}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {400, 404} and _looks_like_model_error(body):
            _probe_chat_completion(base_url=base_url, api_key=api_key, model=fallback_model)
            return fallback_model, {
                "fallback": True,
                "requested_model": requested_model,
                "resolved_model": fallback_model,
                "reason": body,
            }
        raise


def _fetch_model_catalog(*, base_url: str, api_key: str) -> set[str]:
    url = base_url.rstrip("/") + "/models"
    req = request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with request.urlopen(req, timeout=30.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = set()
    for item in payload.get("data", []):
        if isinstance(item, dict) and item.get("id") is not None:
            models.add(str(item["id"]))
    return models


def _probe_chat_completion(*, base_url: str, api_key: str, model: str) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": 'Reply with {"ok": true}.'}],
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=45.0):
        return


def _looks_like_model_error(body: str) -> bool:
    text = body.lower()
    return any(token in text for token in ("unsupported", "unknown model", "model", "not found", "does not exist"))


def _write_run_manifest_note(
    *,
    cache_root: Path,
    preset: str,
    run_name: str,
    config_path: Path,
    resolved_model: str | None,
    fallback_note: dict[str, Any] | None,
) -> None:
    payload = {
        "preset": preset,
        "run_name": run_name,
        "config_path": str(config_path),
        "resolved_model": resolved_model,
        "fallback_note": fallback_note,
    }
    path = cache_root / "configs" / f"{preset}_resolved.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
