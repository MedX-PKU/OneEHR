"""Fairness radar chart for comparing metrics across demographic subgroups."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oneehr.visualization._style import get_palette, save_and_close
from oneehr.visualization._utils import load_analysis_json


def plot_fairness_radar(
    fairness_data: dict | Path,
    *,
    system: str | None = None,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Fairness Metrics",
) -> plt.Figure:
    """Radar (spider) chart of fairness metrics across sensitive attribute groups.

    Parameters
    ----------
    fairness_data : dict or Path
        Loaded fairness.json or a run_dir Path.
    system : str, optional
        System to plot. If None, uses the first system.
    """
    if isinstance(fairness_data, Path):
        fairness_data = load_analysis_json(fairness_data, "fairness")

    systems_list = fairness_data.get("systems", [])
    if not systems_list:
        raise ValueError("No fairness data available")

    if system is None:
        system = systems_list[0]["name"]

    sys_data = next((s for s in systems_list if s["name"] == system), None)
    if sys_data is None:
        raise ValueError(f"System {system!r} not found in fairness data")

    attributes = sys_data.get("attributes", {})
    if not attributes:
        raise ValueError(f"No fairness attributes for system {system!r}")

    from oneehr.visualization._style import apply_style
    preset = apply_style(style)

    # Collect metrics across all attribute groups.
    all_groups: list[str] = []
    all_metrics: dict[str, list[float]] = {}

    for attr_name, attr_data in attributes.items():
        groups = attr_data.get("groups", {})
        for group_name, group_metrics in groups.items():
            label = f"{attr_name}:{group_name}"
            all_groups.append(label)
            for metric_name, val in group_metrics.items():
                if isinstance(val, (int, float)):
                    all_metrics.setdefault(metric_name, []).append(val)

    if not all_metrics:
        raise ValueError("No numeric fairness metrics found")

    # Use metrics that have values for all groups.
    n_groups = len(all_groups)
    metric_names = [
        k for k, v in all_metrics.items()
        if len(v) == n_groups
    ][:8]  # Limit to 8 metrics for readability.

    if not metric_names:
        raise ValueError("No consistent metrics across groups")

    # Radar plot.
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(
        figsize=figsize or (5, 5),
        subplot_kw={"projection": "polar"},
    )
    palette = get_palette(n_groups, style)

    for i, group_label in enumerate(all_groups):
        vals = [all_metrics[m][i] for m in metric_names]
        vals.append(vals[0])  # Close the polygon.
        ax.plot(angles, vals, "o-", color=palette[i], lw=1.2, ms=3,
                label=group_label)
        ax.fill(angles, vals, color=palette[i], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=preset.get("xtick.labelsize", 8))
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              frameon=True, fontsize=preset.get("legend.fontsize", 7))

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
