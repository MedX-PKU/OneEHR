"""Model comparison forest plot with confidence intervals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from oneehr.visualization._style import get_palette, new_figure, save_and_close
from oneehr.visualization._utils import load_analysis_json


def plot_forest(
    comparison: dict | Path,
    *,
    metrics: list[str] | None = None,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Model Comparison",
) -> plt.Figure:
    """Horizontal forest plot of metric point estimates + 95 % CI per system.

    Parameters
    ----------
    comparison : dict or Path
        Either the loaded comparison.json dict or a run_dir Path (reads
        analyze/comparison.json automatically).
    metrics : list[str], optional
        Which metrics to plot.  Defaults to all metrics that have CI bounds.
    """
    if isinstance(comparison, Path):
        comparison = load_analysis_json(comparison, "comparison")

    systems = comparison.get("systems", [])
    if not systems:
        raise ValueError("comparison.json contains no systems")

    # Discover metrics with CIs.
    sample = systems[0].get("metrics", {})
    if metrics is None:
        metrics = [k for k in sample if f"{k}_ci_low" in sample and f"{k}_ci_high" in sample]
    if not metrics:
        metrics = [k for k in sample if isinstance(sample.get(k), (int, float))][:3]

    n_metrics = len(metrics)
    n_systems = len(systems)

    fig_h = max(3.0, 0.4 * n_systems * n_metrics + 1.5)
    fig, ax = new_figure(style=style, figsize=figsize or (6, fig_h), ax=ax)
    palette = get_palette(n_metrics, style)

    y_labels = []
    y_pos = []
    colours = []
    points = []
    lows = []
    highs = []

    pos = 0
    for sys_info in reversed(systems):
        name = sys_info["name"]
        m = sys_info.get("metrics", {})
        for j, metric in enumerate(metrics):
            val = m.get(metric, float("nan"))
            lo = m.get(f"{metric}_ci_low", val)
            hi = m.get(f"{metric}_ci_high", val)
            y_labels.append(f"{name}")
            y_pos.append(pos)
            points.append(val)
            lows.append(val - lo)
            highs.append(hi - val)
            colours.append(palette[j])
            pos += 1
        pos += 0.5  # gap between systems

    ax.barh(
        y_pos,
        points,
        xerr=[lows, highs],
        color=colours,
        height=0.6,
        edgecolor="none",
        capsize=3,
        ecolor="#555555",
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Score")
    ax.set_title(title)

    # Legend for metrics.
    if n_metrics > 1:
        from matplotlib.patches import Patch

        handles = [Patch(color=palette[j], label=metrics[j]) for j in range(n_metrics)]
        ax.legend(handles=handles, loc="lower right", frameon=True)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
