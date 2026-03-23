"""Precision-Recall curve visualization with multi-system overlay and CI."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from oneehr.visualization._style import get_palette, new_figure, save_and_close
from oneehr.visualization._utils import bootstrap_curve, system_predictions


def _pr_xy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    # Return sorted by recall (ascending) for interpolation.
    idx = np.argsort(recall)
    return recall[idx], precision[idx]


def plot_pr(
    predictions: pd.DataFrame | Path,
    *,
    systems: list[str] | None = None,
    ci: bool = True,
    n_boot: int = 200,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Precision-Recall Curves",
) -> plt.Figure:
    """Multi-system PR curves with AUPRC in legend and optional CI shading."""
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_parquet(predictions)

    fig, ax = new_figure(style=style, figsize=figsize, ax=ax)
    names = systems or sorted(predictions["system"].unique())
    palette = get_palette(len(names), style)

    # Compute baseline (prevalence).
    all_true = predictions["y_true"].dropna().to_numpy(dtype=float)
    prevalence = all_true.mean() if len(all_true) > 0 else 0.5

    for i, name in enumerate(names):
        y_true, y_pred = system_predictions(predictions, name)
        if len(y_true) == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        ax.plot(recall, precision, color=palette[i], lw=1.5, label=f"{name} (AP={ap:.3f})")

        if ci:
            x_common, y_low, y_high = bootstrap_curve(
                y_true,
                y_pred,
                _pr_xy,
                n_boot=n_boot,
            )
            ax.fill_between(x_common, y_low, y_high, color=palette[i], alpha=0.12)

    ax.axhline(prevalence, color="k", ls="--", lw=0.8, alpha=0.4, label=f"Baseline ({prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
