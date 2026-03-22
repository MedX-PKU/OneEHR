"""Feature importance bar chart and SHAP beeswarm plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oneehr.visualization._style import get_palette, new_figure, save_and_close
from oneehr.visualization._utils import load_analysis_json


def plot_feature_importance(
    importance_data: dict | Path,
    *,
    model: str | None = None,
    top_n: int = 20,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importance_data : dict or Path
        Loaded feature_importance.json or a run_dir Path.
    model : str, optional
        Which model to plot.  If None, plots the first available model.
    """
    if isinstance(importance_data, Path):
        importance_data = load_analysis_json(importance_data, "feature_importance")

    models_dict = importance_data.get("models", {})
    if not models_dict:
        raise ValueError("No feature importance data available")

    if model is None:
        # Pick first model without error.
        model = next(
            (k for k, v in models_dict.items() if "error" not in v),
            next(iter(models_dict)),
        )

    entry = models_dict[model]
    if "error" in entry:
        raise ValueError(f"Feature importance failed for {model}: {entry['error']}")

    features = np.array(entry["features"])
    importances = np.array(entry["importances"], dtype=float)

    # Sort descending and take top N.
    order = np.argsort(importances)[::-1][:top_n]
    features = features[order][::-1]  # Reverse for horizontal bar (top at top).
    importances = importances[order][::-1]

    fig_h = max(3.0, 0.35 * len(features) + 1.0)
    fig, ax = new_figure(style=style, figsize=figsize or (5, fig_h), ax=ax)
    palette = get_palette(1, style)

    ax.barh(range(len(features)), importances, color=palette[0], edgecolor="none",
            height=0.7, alpha=0.85)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance")
    ax.set_title(title or f"Feature Importance ({model})")

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_data: np.ndarray | None = None,
    *,
    top_n: int = 20,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "SHAP Feature Impact",
) -> plt.Figure:
    """Beeswarm plot of SHAP values.

    Parameters
    ----------
    shap_values : (n_samples, n_features)
        SHAP values matrix.
    feature_names : list[str]
        Feature names corresponding to columns.
    feature_data : (n_samples, n_features), optional
        Raw feature values for coloring dots by feature value.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    # Reverse for display (top feature at top).
    order = order[::-1]

    fig_h = max(3.0, 0.4 * len(order) + 1.0)
    fig, ax = new_figure(style=style, figsize=figsize or (6, fig_h), ax=ax)

    for j_pos, j_feat in enumerate(order):
        vals = shap_values[:, j_feat]
        # Add jitter for visibility.
        jitter = np.random.default_rng(42).normal(0, 0.15, size=len(vals))
        y = np.full_like(vals, j_pos) + jitter

        if feature_data is not None:
            fv = feature_data[:, j_feat]
            fv_norm = fv - np.nanmin(fv)
            denom = np.nanmax(fv) - np.nanmin(fv)
            if denom > 0:
                fv_norm = fv_norm / denom
            else:
                fv_norm = np.full_like(fv_norm, 0.5)
            ax.scatter(vals, y, c=fv_norm, cmap="coolwarm", s=5, alpha=0.6,
                       edgecolors="none", vmin=0, vmax=1)
        else:
            ax.scatter(vals, y, c=vals, cmap="coolwarm", s=5, alpha=0.6,
                       edgecolors="none")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[j] for j in order])
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("SHAP Value")
    ax.set_title(title)

    if feature_data is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(0, 1), cmap="coolwarm")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Feature Value")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
