"""oneehr preprocess subcommand."""
from __future__ import annotations


def run_preprocess(cfg_path: str, *, overview: bool, overview_top_k_codes: int) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.data.io import load_dynamic_table_optional, load_label_table, load_static_table
    from oneehr.data.splits import build_splits_for_dataset, save_splits
    from oneehr.artifacts.materialize import materialize_preprocess_artifacts

    cfg = load_experiment_config(cfg_path)
    dynamic = load_dynamic_table_optional(cfg.dataset.dynamic)
    static = load_static_table(cfg.dataset.static)
    label = load_label_table(cfg.dataset.label)
    out_root = cfg.output.root / cfg.output.run_name
    materialize_preprocess_artifacts(dynamic=dynamic, static=static, label=label, cfg=cfg, out_root=out_root)
    splits = build_splits_for_dataset(
        dynamic=dynamic,
        static=static,
        split=cfg.split,
        repeat=cfg.trainer.repeat,
    )
    save_splits(splits, out_root / "splits")
    if overview:
        import json

        from oneehr.artifacts.read import read_run_manifest
        from oneehr.data.overview_light import build_dataset_overview, build_feature_overview

        if dynamic is None:
            payload = {"note": "dynamic table not provided; overview is limited."}
        else:
            payload = build_dataset_overview(dynamic, cfg.dataset.dynamic, top_k_codes=overview_top_k_codes)
        manifest = read_run_manifest(out_root)
        if manifest is not None:
            payload["features"] = build_feature_overview(
                dynamic_feature_columns=manifest.dynamic_feature_columns(),
                static_feature_columns=manifest.static_feature_columns(),
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
