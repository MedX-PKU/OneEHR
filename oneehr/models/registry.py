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
    uses_static: bool = False


def _standard_kwargs(mc, input_dim, out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        out_dim=out_dim,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    )


def _rnn_kwargs(mc, input_dim, out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        out_dim=out_dim,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
        bidirectional=mc.bidirectional,
        nonlinearity=mc.nonlinearity,
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


def _stagenet_kwargs(mc, input_dim, _out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        conv_size=mc.conv_size,
        levels=mc.levels,
        dropconnect=mc.dropconnect,
        dropout=mc.dropout,
        dropres=mc.dropres,
    )


def _retain_kwargs(mc, input_dim, _out_dim):
    return dict(input_dim=input_dim, hidden_dim=mc.hidden_dim, dropout=mc.dropout)


def _adacare_kwargs(mc, input_dim, _out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        kernel_size=mc.kernel_size,
        kernel_num=mc.kernel_num,
        r_v=mc.r_v,
        r_c=mc.r_c,
        dropout=mc.dropout,
    )


def _concare_kwargs(mc, input_dim, _out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        num_heads=mc.num_heads,
        dropout=mc.dropout,
    )


def _grasp_kwargs(mc, input_dim, _out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        cluster_num=mc.cluster_num,
        dropout=mc.dropout,
    )


def _mlp_kwargs(mc, input_dim, _out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    )


def _dragent_kwargs(mc, input_dim, _out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        dropout=mc.dropout,
    )


def _mcgru_kwargs(mc, input_dim, _out_dim):
    return dict(
        input_dim=input_dim,
        hidden_dim=mc.hidden_dim,
        feat_dim=mc.hidden_dim // 16 if mc.hidden_dim >= 16 else 8,
        dropout=mc.dropout,
    )


_SPECS: dict[str, _ModelSpec] = {
    "gru": _ModelSpec("oneehr.models.gru", "GRUModel", "GRUTimeModel", "gru", _standard_kwargs),
    "lstm": _ModelSpec("oneehr.models.lstm", "LSTMModel", "LSTMTimeModel", "lstm", _standard_kwargs),
    "rnn": _ModelSpec("oneehr.models.rnn", "RNNModel", "RNNTimeModel", "rnn", _rnn_kwargs),
    "transformer": _ModelSpec("oneehr.models.transformer", "TransformerModel", "TransformerTimeModel", "transformer", _transformer_kwargs),
    "tcn": _ModelSpec("oneehr.models.tcn", None, "TCNTimeModel", "tcn", _tcn_kwargs),
    "stagenet": _ModelSpec("oneehr.models.stagenet", "StageNetModel", "StageNetTimeModel", "stagenet", _stagenet_kwargs),
    "retain": _ModelSpec("oneehr.models.retain", "RETAINModel", "RETAINTimeModel", "retain", _retain_kwargs),
    "adacare": _ModelSpec("oneehr.models.adacare", "AdaCareModel", "AdaCareTimeModel", "adacare", _adacare_kwargs),
    "concare": _ModelSpec("oneehr.models.concare", "ConCareModel", "ConCareTimeModel", "concare", _concare_kwargs),
    "grasp": _ModelSpec("oneehr.models.grasp", "GRASPModel", "GRASPTimeModel", "grasp", _grasp_kwargs),
    "mlp": _ModelSpec("oneehr.models.mlp", "MLPModel", "MLPTimeModel", "mlp", _mlp_kwargs),
    "dragent": _ModelSpec("oneehr.models.dragent", "DrAgentModel", "DrAgentTimeModel", "dragent", _dragent_kwargs, uses_static=True),
    "mcgru": _ModelSpec("oneehr.models.mcgru", "MCGRUModel", "MCGRUTimeModel", "mcgru", _mcgru_kwargs, uses_static=True),
}


def build_model(cfg: ExperimentConfig) -> BuiltModel:
    """Build a model instance from config.

    This function centralizes the name->implementation mapping so the CLI and
    other entrypoints don't grow large if/elif ladders.
    """
    name = cfg.model.name
    input_dim = int(getattr(cfg, "_dynamic_dim", 0) or 0)
    out_dim = 1

    # Tabular models don't need instantiation.
    from oneehr.models.constants import TABULAR_MODELS
    if name in TABULAR_MODELS:
        return BuiltModel(model=None, kind="ml")

    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unsupported model.name={name!r}")

    mc = getattr(cfg.model, spec.config_attr)
    kwargs = spec.build_kwargs(mc, input_dim, out_dim)

    if spec.uses_static:
        kwargs["static_dim"] = int(getattr(cfg, "_static_dim", 0) or 0)

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
