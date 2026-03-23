"""Missing data heatmap visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from oneehr.visualization._style import new_figure, save_and_close


def plot_missing_heatmap(
    binned: pd.DataFrame | Path,
    *,
    max_features: int = 40,
    max_patients: int = 200,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Missing Data Pattern",
) -> plt.Figure:
    """Feature x patient missingness heatmap.

    Parameters
    ----------
    binned : DataFrame or Path
        The binned.parquet from preprocessing.
    max_features : int
        Maximum features to show (sorted by missingness rate).
    max_patients : int
        Maximum patients to sample for display.
    """
    if isinstance(binned, (str, Path)):
        binned = pd.read_parquet(binned)

    # Use feature columns only.
    feat_cols = [c for c in binned.columns if c.startswith(("num__", "cat__"))]
    if not feat_cols:
        raise ValueError("No feature columns found in binned data")

    # Compute per-feature missingness rate.
    miss_rate = binned[feat_cols].isna().mean().sort_values(ascending=False)

    # Filter to features with some missingness.
    miss_rate = miss_rate[miss_rate > 0]
    if miss_rate.empty:
        raise ValueError("No missing data found")

    top_feats = miss_rate.index[:max_features].tolist()

    # Patient-level last-observation view.
    last_obs = binned.sort_values(["patient_id", "bin_time"], kind="stable").groupby("patient_id", sort=False)[top_feats].last()

    # Sample patients if too many.
    if len(last_obs) > max_patients:
        last_obs = last_obs.sample(n=max_patients, random_state=42)

    mask = last_obs.isna().astype(int)

    fig_w = max(6, 0.2 * len(top_feats) + 2)
    fig_h = max(4, 0.04 * len(mask) + 2)
    fig, ax = new_figure(style=style, figsize=figsize or (fig_w, fig_h), ax=ax)

    sns.heatmap(
        mask,
        cmap=["#FFFFFF", "#CC3311"],
        cbar_kws={"label": "Missing", "ticks": [0, 1]},
        xticklabels=True,
        yticklabels=False,
        ax=ax,
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel(f"Patient (n={len(mask)})")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_missingness_bar(
    binned: pd.DataFrame | Path,
    *,
    top_n: int = 30,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Feature Missingness Rate",
) -> plt.Figure:
    """Horizontal bar chart of per-feature missingness rate."""
    if isinstance(binned, (str, Path)):
        binned = pd.read_parquet(binned)

    feat_cols = [c for c in binned.columns if c.startswith(("num__", "cat__"))]
    miss_rate = binned[feat_cols].isna().mean().sort_values(ascending=False)
    miss_rate = miss_rate[miss_rate > 0].head(top_n)

    if miss_rate.empty:
        raise ValueError("No missing data found")

    # Reverse for horizontal bar (top at top).
    miss_rate = miss_rate.iloc[::-1]

    fig_h = max(3, 0.35 * len(miss_rate) + 1)
    fig, ax = new_figure(style=style, figsize=figsize or (5, fig_h), ax=ax)

    from oneehr.visualization._style import get_palette

    palette = get_palette(1, style)

    ax.barh(range(len(miss_rate)), miss_rate.values, color=palette[0], edgecolor="none", height=0.7, alpha=0.85)
    ax.set_yticks(range(len(miss_rate)))
    ax.set_yticklabels(miss_rate.index)
    ax.set_xlabel("Missing Rate")
    ax.set_title(title)
    ax.set_xlim(0, min(1.0, miss_rate.max() * 1.15))

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
