"""Feature attribution visualizations: heatmaps and waterfall plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from oneehr.visualization._style import new_figure, save_and_close, get_palette


def plot_attribution_heatmap(
    attributions: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 20,
    title: str = "Feature Attribution Heatmap",
    xlabel: str = "Patient",
    ylabel: str = "Feature",
    cmap: str = "RdBu_r",
    style: str = "default",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a heatmap of per-instance feature attributions.

    Parameters
    ----------
    attributions : (N, D) array of attribution values (e.g., SHAP values)
    feature_names : (D,) feature names
    top_k : number of top features to show (by mean |attribution|)
    """
    fig, main_ax = new_figure(style=style, ax=ax)

    # Select top-k features
    mean_abs = np.abs(attributions).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]

    data = attributions[:, top_idx].T
    labels = [feature_names[i] for i in top_idx]

    vmax = np.abs(data).max()
    im = main_ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    main_ax.set_yticks(range(len(labels)))
    main_ax.set_yticklabels(labels)
    main_ax.set_xlabel(xlabel)
    main_ax.set_ylabel(ylabel)
    main_ax.set_title(title)

    fig.colorbar(im, ax=main_ax, label="Attribution")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_waterfall(
    importances: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 15,
    title: str = "Feature Importance",
    xlabel: str = "Importance",
    style: str = "default",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a horizontal bar chart (waterfall-style) of feature importance.

    Parameters
    ----------
    importances : (D,) array of importance values
    feature_names : (D,) feature names
    top_k : number of top features to show
    """
    fig, main_ax = new_figure(style=style, ax=ax)

    # Sort and select top-k
    idx = np.argsort(np.abs(importances))[-top_k:]
    values = importances[idx]
    names = [feature_names[i] for i in idx]

    colors = get_palette(2, style=style)
    bar_colors = [colors[0] if v >= 0 else colors[1] for v in values]

    main_ax.barh(range(len(names)), values, color=bar_colors)
    main_ax.set_yticks(range(len(names)))
    main_ax.set_yticklabels(names)
    main_ax.set_xlabel(xlabel)
    main_ax.set_title(title)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_attention_over_time(
    attention_weights: np.ndarray,
    *,
    patient_idx: int = 0,
    time_labels: list[str] | None = None,
    title: str = "Attention Weights Over Time",
    style: str = "default",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot attention weights over time steps for a single patient.

    Parameters
    ----------
    attention_weights : (B, T) or (T,) attention weights
    patient_idx : which patient to plot (if batch dimension present)
    """
    fig, main_ax = new_figure(style=style, ax=ax)

    if attention_weights.ndim == 2:
        attn = attention_weights[patient_idx]
    else:
        attn = attention_weights

    T = len(attn)
    x = range(T)
    labels = time_labels if time_labels is not None else [str(i) for i in x]

    colors = get_palette(1, style=style)
    main_ax.bar(x, attn, color=colors[0], alpha=0.8)
    main_ax.set_xticks(x)
    main_ax.set_xticklabels(labels, rotation=45, ha="right")
    main_ax.set_xlabel("Time Step")
    main_ax.set_ylabel("Attention Weight")
    main_ax.set_title(title)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
