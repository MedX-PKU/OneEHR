"""oneehr preprocess subcommand."""

from __future__ import annotations


def run_preprocess(cfg_path: str) -> None:
    from oneehr.artifacts.materialize import materialize_preprocess_artifacts
    from oneehr.config.load import load_experiment_config
    from oneehr.data.io import load_dynamic_table_optional, load_label_table, load_static_table

    cfg = load_experiment_config(cfg_path)
    dynamic = load_dynamic_table_optional(cfg.dataset.dynamic)
    static = load_static_table(cfg.dataset.static)
    label = load_label_table(cfg.dataset.label)
    run_dir = cfg.run_dir()

    materialize_preprocess_artifacts(
        dynamic=dynamic,
        static=static,
        label=label,
        cfg=cfg,
        run_dir=run_dir,
    )
    print(f"Preprocessed artifacts written to {run_dir / 'preprocess'}")
