"""Pairwise statistical significance heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from oneehr.visualization._style import new_figure, save_and_close
from oneehr.visualization._utils import load_analysis_json


def plot_significance_matrix(
    stats_data: dict | Path,
    *,
    test: str = "delong",
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Pairwise Statistical Significance",
) -> plt.Figure:
    """Heatmap of pairwise p-values between systems.

    Parameters
    ----------
    stats_data : dict or Path
        Loaded statistical_tests.json or a run_dir Path.
    test : str
        Which test to display: "delong" or "mcnemar".
    """
    if isinstance(stats_data, Path):
        stats_data = load_analysis_json(stats_data, "statistical_tests")

    pairwise = stats_data.get("pairwise", [])
    if not pairwise:
        raise ValueError("No pairwise test results found")

    # Collect unique system names.
    systems = sorted(set(
        s for p in pairwise for s in (p["system_a"], p["system_b"])
    ))
    n = len(systems)
    idx = {s: i for i, s in enumerate(systems)}

    # Build p-value matrix.
    matrix = np.ones((n, n))
    for p in pairwise:
        a, b = p["system_a"], p["system_b"]
        test_data = p.get(test, {})
        pval = test_data.get("p_value", 1.0)
        if pval is not None:
            matrix[idx[a], idx[b]] = pval
            matrix[idx[b], idx[a]] = pval

    # Diagonal = NaN (self-comparison).
    np.fill_diagonal(matrix, np.nan)

    sz = max(4, 0.8 * n + 1.5)
    fig, ax = new_figure(style=style, figsize=figsize or (sz, sz), ax=ax)

    # Annotation: show p-value and significance stars.
    annot = np.empty_like(matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = ""
            else:
                p = matrix[i, j]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                annot[i, j] = f"{p:.3f}\n{stars}"

    sns.heatmap(
        matrix, annot=annot, fmt="",
        xticklabels=systems, yticklabels=systems,
        cmap="RdYlGn_r", vmin=0, vmax=0.1,
        mask=np.eye(n, dtype=bool),
        cbar_kws={"label": "p-value"},
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_title(f"{title} ({test})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
