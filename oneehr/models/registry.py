from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from oneehr.config.schema import ExperimentConfig


@dataclass(frozen=True)
class BuiltModel:
    model: object
    kind: str  # dl | ml


@dataclass(frozen=True)
class _ModelSpec:
    module: str
    patient_cls: str
    time_cls: str | None
    config_attr: str
    build_kwargs: Callable  # (model_sub_cfg, input_dim, out_dim) -> dict


def _standard_kwargs(mc, input_dim, out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        out_dim=out_dim,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    )


def _transformer_kwargs(mc, input_dim, out_dim):
    return dict(
        input_dim=input_dim,
        d_model=mc.d_model,
        out_dim=out_dim,
        nhead=mc.nhead,
        num_layers=mc.num_layers,
        dim_feedforward=mc.dim_feedforward,
        dropout=mc.dropout,
    )


def _transformer_patient_kwargs(mc, input_dim, out_dim):
    kw = _transformer_kwargs(mc, input_dim, out_dim)
    kw["pooling"] = mc.pooling
    return kw


def _tcn_kwargs(mc, input_dim, out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        out_dim=out_dim,
        num_layers=mc.num_layers,
        kernel_size=mc.kernel_size,
        dropout=mc.dropout,
    )


_SPECS: dict[str, _ModelSpec] = {
    "gru": _ModelSpec("oneehr.models.gru", "GRUModel", "GRUTimeModel", "gru", _standard_kwargs),
    "lstm": _ModelSpec("oneehr.models.lstm", "LSTMModel", "LSTMTimeModel", "lstm", _standard_kwargs),
    "transformer": _ModelSpec("oneehr.models.transformer", "TransformerModel", "TransformerTimeModel", "transformer", _transformer_kwargs),
    "tcn": _ModelSpec("oneehr.models.tcn", "TCNPatientModel", "TCNTimeModel", "tcn", _tcn_kwargs),
}


def build_model(cfg: ExperimentConfig) -> BuiltModel:
    """Build a model instance from config."""
    primary_model = cfg.require_model(context="model construction")
    name = primary_model.name
    input_dim = int(getattr(cfg, "_dynamic_dim", 0) or 0)
    out_dim = 1

    from oneehr.models.constants import TABULAR_MODELS
    if name in TABULAR_MODELS:
        return BuiltModel(model=None, kind="ml")

    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unsupported model.name={name!r}")

    mc = getattr(primary_model, spec.config_attr)
    kwargs = spec.build_kwargs(mc, input_dim, out_dim)

    is_time = cfg.task.prediction_mode == "time"

    if is_time:
        cls_name = spec.time_cls
        if cls_name is None:
            raise ValueError(f"model.name={name!r} does not support prediction_mode='time'")
    else:
        cls_name = spec.patient_cls
        if cls_name is None:
            raise ValueError(f"model.name={name!r} currently supports prediction_mode='time' only")

    # Special handling: TransformerModel accepts `pooling` but TransformerTimeModel does not.
    if name == "transformer" and not is_time:
        kwargs = _transformer_patient_kwargs(mc, input_dim, out_dim)

    mod = import_module(spec.module)
    cls = getattr(mod, cls_name)
    return BuiltModel(model=cls(**kwargs), kind="dl")
