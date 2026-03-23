"""Decision Curve Analysis (DCA) for clinical utility assessment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oneehr.visualization._style import get_palette, new_figure, save_and_close
from oneehr.visualization._utils import system_predictions


def _net_benefit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> float:
    """Compute net benefit at a given threshold."""
    n = len(y_true)
    if n == 0:
        return 0.0
    y_hat = (y_pred >= threshold).astype(int)
    tp = ((y_hat == 1) & (y_true == 1)).sum()
    fp = ((y_hat == 1) & (y_true == 0)).sum()
    odds = threshold / (1 - threshold) if threshold < 1.0 else float("inf")
    return (tp / n) - (fp / n) * odds


def plot_decision_curve(
    predictions: pd.DataFrame | Path,
    *,
    systems: list[str] | None = None,
    thresholds: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Decision Curve Analysis",
) -> plt.Figure:
    """Decision Curve Analysis: net benefit vs threshold probability.

    Includes "Treat All" and "Treat None" reference lines.
    """
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_parquet(predictions)

    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    fig, ax = new_figure(style=style, figsize=figsize, ax=ax)
    names = systems or sorted(predictions["system"].unique())
    palette = get_palette(len(names) + 1, style)

    # "Treat All" reference.
    all_true = predictions.drop_duplicates("patient_id")["y_true"].dropna().to_numpy(float)
    prevalence = all_true.mean() if len(all_true) > 0 else 0.5
    treat_all_nb = [prevalence - (1 - prevalence) * (t / (1 - t)) if t < 1 else 0 for t in thresholds]
    ax.plot(thresholds, treat_all_nb, "k-", lw=1, alpha=0.5, label="Treat All")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.4, label="Treat None")

    for i, name in enumerate(names):
        y_true, y_pred = system_predictions(predictions, name)
        if len(y_true) == 0:
            continue
        nb = [_net_benefit(y_true, y_pred, t) for t in thresholds]
        ax.plot(thresholds, nb, color=palette[i], lw=1.5, label=name)

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.set_xlim(0, 1)
    y_vals = ax.get_ylim()
    ax.set_ylim(max(y_vals[0], -0.1), y_vals[1])

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
