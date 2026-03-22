"""oneehr plot subcommand.

Renders publication-quality figures from a completed run's artifacts.
"""
from __future__ import annotations

from pathlib import Path


def run_plot(
    cfg_path: str,
    *,
    figures: list[str] | None = None,
    style: str = "default",
    output: str | None = None,
) -> None:
    from oneehr.config.load import load_experiment_config
    from oneehr.visualization import discover_available, list_figures, render_figure

    cfg = load_experiment_config(cfg_path)
    run_dir = cfg.run_dir()

    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    save_dir = Path(output) if output else run_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    available = discover_available(run_dir)
    if figures:
        all_known = set(list_figures())
        for f in figures:
            if f not in all_known:
                raise SystemExit(
                    f"Unknown figure: {f!r}. Available: {sorted(all_known)}"
                )
        to_render = [f for f in figures if f in available]
        skipped = [f for f in figures if f not in available]
        if skipped:
            print(f"Skipping (missing artifacts): {', '.join(skipped)}")
    else:
        to_render = available

    if not to_render:
        print("No figures can be rendered (missing required artifacts).")
        print("Run `oneehr test` and `oneehr analyze` first.")
        return

    print(f"Rendering {len(to_render)} figure(s) with style={style!r}")
    print(f"Output: {save_dir}")

    for name in to_render:
        print(f"  {name}...", end=" ", flush=True)
        try:
            render_figure(name, run_dir=run_dir, style=style, save_dir=save_dir)
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")

    print(f"Figures saved to {save_dir}")
