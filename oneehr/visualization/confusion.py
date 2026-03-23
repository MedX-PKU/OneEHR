"""Confusion matrix heatmap visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from oneehr.visualization._style import new_figure, save_and_close
from oneehr.visualization._utils import system_predictions


def plot_confusion_matrix(
    predictions: pd.DataFrame | Path,
    *,
    system: str | None = None,
    threshold: float = 0.5,
    normalize: bool = False,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Annotated confusion matrix heatmap for a single system.

    Parameters
    ----------
    system : str, optional
        System to plot.  If None, uses the first system.
    normalize : bool
        If True, shows row-normalized proportions.
    """
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_parquet(predictions)

    if system is None:
        system = sorted(predictions["system"].unique())[0]

    y_true, y_pred = system_predictions(predictions, system)
    y_hat = (y_pred >= threshold).astype(int)

    from sklearn.metrics import confusion_matrix as _cm

    cm = _cm(y_true.astype(int), y_hat, labels=[0, 1])

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = new_figure(style=style, figsize=figsize or (4, 3.5), ax=ax)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar=True,
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or f"Confusion Matrix ({system})")

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_confusion_grid(
    predictions: pd.DataFrame | Path,
    *,
    systems: list[str] | None = None,
    threshold: float = 0.5,
    normalize: bool = False,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrices",
) -> plt.Figure:
    """Side-by-side confusion matrices for multiple systems."""
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_parquet(predictions)

    names = systems or sorted(predictions["system"].unique())
    n = len(names)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    from oneehr.visualization._style import apply_style

    preset = apply_style(style)

    per_w = 3.0
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (per_w * ncols, per_w * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        plot_confusion_matrix(
            predictions,
            system=name,
            threshold=threshold,
            normalize=normalize,
            ax=axes[r, c],
            style=style,
            title=name,
        )

    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=preset.get("axes.titlesize", 12))
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
