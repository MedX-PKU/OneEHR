"""Calibration (reliability) diagram with ECE annotation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oneehr.visualization._style import get_palette, new_figure, save_and_close
from oneehr.visualization._utils import system_predictions


def _reliability_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute reliability diagram data and ECE."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred > lo) & (y_pred <= hi)
        if lo == 0.0:
            mask |= y_pred == 0.0
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = y_pred[mask].mean()
        avg_acc = y_true[mask].mean()
        bin_centers.append(avg_conf)
        bin_accs.append(avg_acc)
        bin_counts.append(count)
        ece += (count / n) * abs(avg_acc - avg_conf)

    return (
        np.array(bin_centers),
        np.array(bin_accs),
        np.array(bin_counts),
        ece,
    )


def plot_calibration(
    predictions: pd.DataFrame | Path,
    *,
    systems: list[str] | None = None,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Calibration Plot",
) -> plt.Figure:
    """Reliability diagram (predicted vs observed probability) per system."""
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_parquet(predictions)

    fig, ax = new_figure(style=style, figsize=figsize, ax=ax)
    names = systems or sorted(predictions["system"].unique())
    palette = get_palette(len(names), style)

    for i, name in enumerate(names):
        y_true, y_pred = system_predictions(predictions, name)
        if len(y_true) == 0:
            continue

        centers, accs, counts, ece = _reliability_data(y_true, y_pred, n_bins)
        ax.plot(centers, accs, "o-", color=palette[i], lw=1.5, ms=4,
                label=f"{name} (ECE={ece:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
