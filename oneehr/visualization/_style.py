"""Publication-quality style presets for OneEHR visualizations.

Provides journal-specific rcParams, colorblind-safe palettes, and helper
functions for consistent figure styling across all visualization modules.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colorblind-safe palettes
# ---------------------------------------------------------------------------

# Paul Tol's vibrant palette (7 colours, excellent for colour-blind readers).
_PALETTE_VIBRANT = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
]

# Extended palette (12 colours) using Paul Tol's muted scheme.
_PALETTE_MUTED = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#DDDDDD",  # pale grey
    "#661100",  # brown
    "#6699CC",  # light blue
]

# ---------------------------------------------------------------------------
# Journal style presets
# ---------------------------------------------------------------------------

JOURNAL_STYLES: dict[str, dict[str, Any]] = {
    "default": {
        "figsize": (6, 5),
        "dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "palette": _PALETTE_VIBRANT,
    },
    "nature": {
        # Nature single-column = 89 mm ≈ 3.50 in
        "figsize": (3.50, 3.0),
        "dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "palette": _PALETTE_VIBRANT,
    },
    "lancet": {
        "figsize": (3.35, 3.0),
        "dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "palette": _PALETTE_VIBRANT,
    },
    "wide": {
        # Nature double-column = 183 mm ≈ 7.20 in
        "figsize": (7.20, 4.0),
        "dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "palette": _PALETTE_VIBRANT,
    },
}

# Keys that are NOT rcParams (handled separately).
_NON_RC_KEYS = {"figsize", "palette"}


def apply_style(style: str = "default") -> dict[str, Any]:
    """Apply journal-specific matplotlib rcParams.

    Returns the full style dict (including non-rcParam keys like figsize,
    palette) so callers can access them.
    """
    preset = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
    rc = {k: v for k, v in preset.items() if k not in _NON_RC_KEYS}
    mpl.rcParams.update(rc)
    return preset


def get_palette(n: int = 7, style: str = "default") -> list[str]:
    """Return *n* colorblind-safe colours for the given style."""
    preset = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
    base = preset.get("palette", _PALETTE_VIBRANT)
    if n <= len(base):
        return base[:n]
    # Cycle if more colours needed.
    return [base[i % len(base)] for i in range(n)]


def get_figsize(style: str = "default") -> tuple[float, float]:
    """Return default (width, height) for the given style."""
    preset = JOURNAL_STYLES.get(style, JOURNAL_STYLES["default"])
    return preset["figsize"]


def new_figure(
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create or reuse a figure/axes pair with the given style applied.

    If *ax* is provided the existing figure is returned.  Otherwise a fresh
    figure is created with the style's default figsize (overridable).
    """
    preset = apply_style(style)
    if ax is not None:
        return ax.figure, ax
    sz = figsize or preset["figsize"]
    fig, ax = plt.subplots(figsize=sz)
    return fig, ax


def save_and_close(
    fig: plt.Figure,
    save_path: str | None,
    *,
    dpi: int = 300,
    tight: bool = True,
) -> None:
    """Save a figure to disk (PNG + PDF) and close it."""
    if save_path is None:
        return
    from pathlib import Path

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(p, dpi=dpi, bbox_inches=bbox)
    # Also save PDF for vector output.
    fig.savefig(p.with_suffix(".pdf"), bbox_inches=bbox)
    plt.close(fig)
