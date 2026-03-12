from __future__ import annotations

import pandas as pd

from oneehr.config.schema import ExperimentConfig
from oneehr.llm.templates import get_prompt_template


def render_prompt(
    *,
    cfg: ExperimentConfig,
    instance: dict[str, object],
    dynamic: pd.DataFrame | None,
    static_row: pd.Series | None,
    schema_text: str,
    template_name: str | None = None,
    prompt_cfg=None,
    target_prediction: dict[str, object] | None = None,
    analysis_refs: dict[str, object] | None = None,
) -> str:
    name = str(template_name or cfg.llm.prompt_template)
    spec = get_prompt_template(name)
    renderer = spec.renderer
    if renderer is None:  # pragma: no cover
        raise ValueError(f"Prompt template {name!r} is missing a renderer.")

    kwargs = {
        "cfg": cfg,
        "instance": instance,
        "dynamic": dynamic,
        "static_row": static_row,
        "schema_text": schema_text,
    }
    if prompt_cfg is not None:
        kwargs["prompt_cfg"] = prompt_cfg
    if target_prediction is not None:
        kwargs["target_prediction"] = target_prediction
    if analysis_refs is not None:
        kwargs["analysis_refs"] = analysis_refs
    return renderer(**kwargs)
