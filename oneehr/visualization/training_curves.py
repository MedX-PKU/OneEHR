"""Training loss and metric curves over epochs for DL models."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oneehr.visualization._style import get_palette, new_figure, save_and_close
from oneehr.visualization._utils import load_training_meta


def plot_training_curves(
    meta: dict | Path,
    *,
    model_name: str | None = None,
    metrics: list[str] | None = None,
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot per-epoch training and validation loss/metrics.

    Parameters
    ----------
    meta : dict or Path
        Either the loaded meta.json dict, or a run_dir Path.  When a
        run_dir is given, *model_name* must be provided.
    """
    if isinstance(meta, Path):
        if model_name is None:
            raise ValueError("model_name required when meta is a run_dir Path")
        meta = load_training_meta(meta, model_name)

    train_metrics = meta.get("train_metrics", {})
    history = train_metrics.get("history", {})
    if not history:
        raise ValueError("No training history found in meta.json")

    # Filter to requested metrics.
    available_keys = sorted(history.keys())
    if metrics is not None:
        plot_keys = [k for k in metrics if k in history]
    else:
        plot_keys = available_keys

    if not plot_keys:
        raise ValueError(f"No matching metrics.  Available: {available_keys}")

    # Group: pair train/val versions of the same metric.
    groups: dict[str, list[str]] = {}
    for k in plot_keys:
        base = k.replace("val_", "").replace("train_", "")
        groups.setdefault(base, []).append(k)

    n_plots = len(groups)
    fig, axes = plt.subplots(1, n_plots,
                             figsize=figsize or (4 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]

    from oneehr.visualization._style import apply_style
    apply_style(style)
    palette = get_palette(2, style)

    for ax_i, (base_name, keys) in zip(axes, groups.items()):
        for j, k in enumerate(sorted(keys)):
            vals = history[k]
            epochs = list(range(1, len(vals) + 1))
            is_val = k.startswith("val_")
            ax_i.plot(
                epochs, vals,
                color=palette[1] if is_val else palette[0],
                lw=1.5,
                ls="--" if is_val else "-",
                label=k,
            )
        ax_i.set_xlabel("Epoch")
        ax_i.set_ylabel(base_name.replace("_", " ").title())
        ax_i.legend(frameon=True, fontsize="small")

    model_label = model_name or meta.get("model_name", "")
    fig.suptitle(title or f"Training Curves ({model_label})")
    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig


def plot_training_curves_multi(
    run_dir: Path,
    *,
    model_names: list[str] | None = None,
    metric: str = "loss",
    ax: plt.Axes | None = None,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """Overlay a single metric across multiple models on one plot."""
    from oneehr.visualization._style import apply_style
    apply_style(style)

    train_dir = run_dir / "train"
    if model_names is None:
        model_names = sorted(
            d.name for d in train_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )

    fig, ax = new_figure(style=style, figsize=figsize, ax=ax)
    palette = get_palette(len(model_names), style)

    for i, mname in enumerate(model_names):
        try:
            meta = load_training_meta(run_dir, mname)
        except FileNotFoundError:
            continue
        history = meta.get("train_metrics", {}).get("history", {})
        # Try val_ version first.
        key = f"val_{metric}" if f"val_{metric}" in history else metric
        if key not in history:
            continue
        vals = history[key]
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals, color=palette[i], lw=1.5, label=mname)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(frameon=True)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
