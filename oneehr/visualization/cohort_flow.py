"""CONSORT-style cohort flow diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from oneehr.visualization._style import save_and_close
from oneehr.visualization._utils import load_split


def plot_cohort_flow(
    run_dir: Path,
    *,
    style: str = "default",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Cohort Flow",
) -> plt.Figure:
    """CONSORT-style flow diagram showing patient split and label distribution.

    Reads split.json and labels.parquet from the run directory.
    """
    import pandas as pd

    from oneehr.visualization._style import apply_style

    preset = apply_style(style)

    split = load_split(run_dir)
    labels_path = run_dir / "preprocess" / "labels.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.exists() else None

    train_ids = set(split.get("train", []))
    val_ids = set(split.get("val", []))
    test_ids = set(split.get("test", []))
    total = len(train_ids | val_ids | test_ids)

    # Count labels per split.
    def _label_info(ids: set) -> str:
        if labels is None:
            return ""
        sub = labels[labels["patient_id"].astype(str).isin(ids)]
        if sub.empty:
            return ""
        pos = (sub["label_value"] == 1).sum()
        neg = len(sub) - pos
        return f"\n(+{pos} / -{neg})"

    fig, ax = plt.subplots(figsize=figsize or (5, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_style = dict(
        boxstyle="round,pad=0.4",
        facecolor="#E8F0FE",
        edgecolor="#4A86C8",
        linewidth=1.2,
    )
    split_style = dict(
        boxstyle="round,pad=0.3",
        facecolor="#F0F8E8",
        edgecolor="#6AAB4A",
        linewidth=1.0,
    )

    # Total cohort box.
    ax.text(5, 9, f"Total Cohort\nn={total}", ha="center", va="center", fontsize=preset.get("font.size", 10), fontweight="bold", bbox=box_style)

    # Arrow from total to split.
    ax.annotate("", xy=(5, 7.6), xytext=(5, 8.3), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555"))

    # Split boxes.
    positions = [(1.5, 6.5), (5, 6.5), (8.5, 6.5)]
    split_data = [
        ("Train", train_ids),
        ("Validation", val_ids),
        ("Test", test_ids),
    ]

    for (x, y), (label, ids) in zip(positions, split_data):
        n = len(ids)
        info = _label_info(ids)
        ax.text(x, y, f"{label}\nn={n}{info}", ha="center", va="center", fontsize=preset.get("font.size", 10) - 1, bbox=split_style)
        ax.annotate("", xy=(x, y + 0.6), xytext=(5, 7.6), arrowprops=dict(arrowstyle="->", lw=0.8, color="#888"))

    ax.set_title(title, fontsize=preset.get("axes.titlesize", 12), pad=10)

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
