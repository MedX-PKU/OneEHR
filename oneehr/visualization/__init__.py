"""OneEHR publication-quality visualization module.

All plot functions follow a consistent interface:
- Return ``matplotlib.figure.Figure``
- Accept optional ``ax`` for subplot embedding
- Accept ``style`` for journal presets ("default", "nature", "lancet", "wide")
- Accept ``save_path`` for direct PNG+PDF export

Quick start::

    from oneehr.visualization import plot_roc, plot_pr
    fig = plot_roc("runs/my_run/test/predictions.parquet")
    fig = plot_pr("runs/my_run/test/predictions.parquet", style="nature")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oneehr.visualization._utils import load_analysis_json
from oneehr.visualization.calibration_plot import plot_calibration
from oneehr.visualization.cohort_flow import plot_cohort_flow
from oneehr.visualization.confusion import plot_confusion_grid, plot_confusion_matrix
from oneehr.visualization.decision_curve import plot_decision_curve
from oneehr.visualization.fairness_plot import plot_fairness_radar
from oneehr.visualization.forest import plot_forest
from oneehr.visualization.importance import plot_feature_importance, plot_shap_beeswarm
from oneehr.visualization.missing_heatmap import plot_missing_heatmap, plot_missingness_bar
from oneehr.visualization.pr import plot_pr
from oneehr.visualization.roc import plot_roc
from oneehr.visualization.significance import plot_significance_matrix
from oneehr.visualization.training_curves import (
    plot_training_curves,
    plot_training_curves_multi,
)

__all__ = [
    # Tier 1
    "plot_roc",
    "plot_pr",
    "plot_forest",
    "plot_calibration",
    "plot_feature_importance",
    "plot_shap_beeswarm",
    "plot_confusion_matrix",
    "plot_confusion_grid",
    "plot_training_curves",
    "plot_training_curves_multi",
    # Tier 2
    "plot_fairness_radar",
    "plot_missing_heatmap",
    "plot_missingness_bar",
    "plot_decision_curve",
    "plot_significance_matrix",
    "plot_cohort_flow",
    # API
    "render_figure",
    "list_figures",
]

# ---------------------------------------------------------------------------
# Registry: maps figure name → (render_fn, required_artifact) so that the
# CLI can auto-discover which figures are plottable for a given run.
# ---------------------------------------------------------------------------

_FIGURE_REGISTRY: dict[str, dict[str, Any]] = {
    "roc": {
        "fn": "_render_roc",
        "requires": "test/predictions.parquet",
        "description": "ROC curves with AUC and bootstrap CI",
    },
    "pr": {
        "fn": "_render_pr",
        "requires": "test/predictions.parquet",
        "description": "Precision-Recall curves with AUPRC",
    },
    "forest": {
        "fn": "_render_forest",
        "requires": "analyze/comparison.json",
        "description": "Model comparison forest plot with CI",
    },
    "calibration": {
        "fn": "_render_calibration",
        "requires": "test/predictions.parquet",
        "description": "Reliability diagram with ECE",
    },
    "feature_importance": {
        "fn": "_render_feature_importance",
        "requires": "analyze/feature_importance.json",
        "description": "Top-N feature importance bar chart",
    },
    "confusion": {
        "fn": "_render_confusion",
        "requires": "test/predictions.parquet",
        "description": "Confusion matrix heatmap per model",
    },
    "training_curves": {
        "fn": "_render_training_curves",
        "requires": "train",
        "description": "Loss/metric curves over epochs",
    },
    # Tier 2
    "fairness": {
        "fn": "_render_fairness",
        "requires": "analyze/fairness.json",
        "description": "Fairness radar chart across subgroups",
    },
    "missing_data": {
        "fn": "_render_missing_data",
        "requires": "preprocess/binned.parquet",
        "description": "Missing data heatmap and bar chart",
    },
    "decision_curve": {
        "fn": "_render_decision_curve",
        "requires": "test/predictions.parquet",
        "description": "Decision curve analysis (net benefit)",
    },
    "significance": {
        "fn": "_render_significance",
        "requires": "analyze/statistical_tests.json",
        "description": "Pairwise p-value significance matrix",
    },
    "cohort_flow": {
        "fn": "_render_cohort_flow",
        "requires": "preprocess/split.json",
        "description": "CONSORT-style cohort flow diagram",
    },
}


def list_figures() -> dict[str, str]:
    """Return ``{name: description}`` for all registered figure types."""
    return {k: v["description"] for k, v in _FIGURE_REGISTRY.items()}


def discover_available(run_dir: Path) -> list[str]:
    """Return figure names whose required artifacts exist in *run_dir*."""
    available = []
    for name, info in _FIGURE_REGISTRY.items():
        req = run_dir / info["requires"]
        if req.exists():
            available.append(name)
    return available


def render_figure(
    name: str,
    *,
    run_dir: Path,
    style: str = "default",
    save_dir: Path | None = None,
    **kwargs: Any,
) -> None:
    """Render a named figure and save to *save_dir*."""
    if name not in _FIGURE_REGISTRY:
        raise ValueError(f"Unknown figure {name!r}. Available: {sorted(_FIGURE_REGISTRY)}")
    entry = _FIGURE_REGISTRY[name]
    fn_name = entry["fn"]
    fn = globals()[fn_name]

    out = save_dir or (run_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)
    fn(run_dir=run_dir, style=style, save_dir=out, **kwargs)


# ---------------------------------------------------------------------------
# Internal render wrappers (called by render_figure).
# ---------------------------------------------------------------------------


def _render_roc(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    preds = run_dir / "test" / "predictions.parquet"
    plot_roc(preds, style=style, save_path=save_dir / "roc.png", **kw)


def _render_pr(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    preds = run_dir / "test" / "predictions.parquet"
    plot_pr(preds, style=style, save_path=save_dir / "pr.png", **kw)


def _render_forest(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    plot_forest(run_dir, style=style, save_path=save_dir / "forest.png", **kw)


def _render_calibration(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    preds = run_dir / "test" / "predictions.parquet"
    plot_calibration(preds, style=style, save_path=save_dir / "calibration.png", **kw)


def _render_feature_importance(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    import json

    fi_path = run_dir / "analyze" / "feature_importance.json"
    fi = json.loads(fi_path.read_text(encoding="utf-8"))
    models = fi.get("models", {})
    for model_name, entry in models.items():
        if "error" in entry:
            continue
        plot_feature_importance(
            fi,
            model=model_name,
            style=style,
            save_path=save_dir / f"feature_importance_{model_name}.png",
            **kw,
        )


def _render_confusion(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    preds = run_dir / "test" / "predictions.parquet"
    plot_confusion_grid(preds, style=style, save_path=save_dir / "confusion.png", **kw)


def _render_training_curves(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    train_dir = run_dir / "train"
    if not train_dir.exists():
        return
    for model_dir in sorted(train_dir.iterdir()):
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            continue
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("train_metrics", {}).get("history"):
            continue
        plot_training_curves(
            meta,
            model_name=model_dir.name,
            style=style,
            save_path=save_dir / f"training_{model_dir.name}.png",
            **kw,
        )


def _render_fairness(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    fairness_data = load_analysis_json(run_dir, "fairness")
    systems = fairness_data.get("systems", [])
    for sys_info in systems:
        name = sys_info["name"]
        plot_fairness_radar(
            fairness_data,
            system=name,
            style=style,
            save_path=save_dir / f"fairness_{name}.png",
            **kw,
        )


def _render_missing_data(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    binned = run_dir / "preprocess" / "binned.parquet"
    try:
        plot_missingness_bar(binned, style=style, save_path=save_dir / "missingness_bar.png", **kw)
    except ValueError:
        pass  # No missing data.
    try:
        plot_missing_heatmap(binned, style=style, save_path=save_dir / "missing_heatmap.png", **kw)
    except ValueError:
        pass


def _render_decision_curve(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    preds = run_dir / "test" / "predictions.parquet"
    plot_decision_curve(preds, style=style, save_path=save_dir / "decision_curve.png", **kw)


def _render_significance(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    stats_data = load_analysis_json(run_dir, "statistical_tests")
    plot_significance_matrix(
        stats_data,
        style=style,
        save_path=save_dir / "significance.png",
        **kw,
    )


def _render_cohort_flow(*, run_dir: Path, style: str, save_dir: Path, **kw: Any) -> None:
    plot_cohort_flow(run_dir, style=style, save_path=save_dir / "cohort_flow.png", **kw)
