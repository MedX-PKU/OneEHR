"""Kaplan-Meier survival curve visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from oneehr.visualization._style import new_figure, save_and_close, get_palette


def plot_kaplan_meier(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    groups: np.ndarray | None = None,
    group_labels: list[str] | None = None,
    *,
    ci: bool = True,
    at_risk: bool = True,
    title: str = "Kaplan-Meier Survival Curve",
    xlabel: str = "Time",
    ylabel: str = "Survival Probability",
    style: str = "default",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Kaplan-Meier survival curves.

    Parameters
    ----------
    event_times : (N,) observed times
    event_observed : (N,) event indicator (1 = event, 0 = censored)
    groups : (N,) optional group labels for stratified curves
    group_labels : optional names for each group
    ci : bool
        Show 95% confidence intervals (Greenwood's formula).
    at_risk : bool
        Show number at risk table below the plot.
    """
    fig, main_ax = new_figure(style=style, ax=ax)

    if groups is None:
        groups = np.zeros(len(event_times), dtype=int)
        group_labels = group_labels or ["All"]

    unique_groups = np.unique(groups)
    if group_labels is None:
        group_labels = [str(g) for g in unique_groups]

    colors = get_palette(len(unique_groups), style=style)
    at_risk_data = {}

    for idx, (g, label, color) in enumerate(zip(unique_groups, group_labels, colors)):
        mask = groups == g
        t = event_times[mask]
        e = event_observed[mask]

        # Sort by time
        order = np.argsort(t)
        t = t[order]
        e = e[order]

        # Kaplan-Meier estimator
        unique_times = np.unique(t[e == 1])
        surv = np.ones(len(unique_times) + 1)
        time_points = np.concatenate([[0], unique_times])
        var_sum = 0.0
        ci_lower = np.ones(len(unique_times) + 1)
        ci_upper = np.ones(len(unique_times) + 1)

        n_at_risk = len(t)
        for i, ti in enumerate(unique_times):
            d = ((t == ti) & (e == 1)).sum()  # events at ti
            c = ((t == ti) & (e == 0)).sum()  # censored at ti
            if n_at_risk > 0:
                surv[i + 1] = surv[i] * (1 - d / n_at_risk)
                if d > 0 and n_at_risk > d:
                    var_sum += d / (n_at_risk * (n_at_risk - d))
            else:
                surv[i + 1] = surv[i]

            # Greenwood's formula
            se = surv[i + 1] * np.sqrt(var_sum) if var_sum > 0 else 0
            ci_lower[i + 1] = max(0, surv[i + 1] - 1.96 * se)
            ci_upper[i + 1] = min(1, surv[i + 1] + 1.96 * se)

            n_at_risk -= d + c
            # Also subtract censored before next event
            if i < len(unique_times) - 1:
                between = ((t > ti) & (t < unique_times[i + 1]) & (e == 0)).sum()
                n_at_risk -= between

        # Step plot
        main_ax.step(time_points, surv, where="post", color=color, label=label, linewidth=1.5)

        if ci:
            main_ax.fill_between(
                time_points, ci_lower, ci_upper,
                step="post", alpha=0.15, color=color,
            )

        # Mark censored observations
        censor_mask = e == 0
        if censor_mask.any():
            censor_times = t[censor_mask]
            # Find survival probability at each censored time
            censor_surv = np.interp(censor_times, time_points, surv)
            main_ax.scatter(
                censor_times, censor_surv,
                marker="|", color=color, s=20, alpha=0.6, zorder=5,
            )

        at_risk_data[label] = (time_points, surv, len(t[mask] if isinstance(mask, np.ndarray) else t))

    main_ax.set_xlabel(xlabel)
    main_ax.set_ylabel(ylabel)
    main_ax.set_title(title)
    main_ax.set_ylim(-0.05, 1.05)
    main_ax.legend(loc="best")

    fig.tight_layout()
    save_and_close(fig, save_path)
    return fig
