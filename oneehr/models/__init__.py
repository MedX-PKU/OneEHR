"""Model architectures and registry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from oneehr.config.schema import ModelConfig, TaskConfig

TABULAR_MODELS: frozenset[str] = frozenset({"xgboost", "catboost"})
DL_MODELS: frozenset[str] = frozenset({"gru", "lstm", "tcn", "transformer"})


@dataclass(frozen=True)
class BuiltModel:
    model: object
    kind: str  # dl | ml


# Default hyperparams per DL model type
_DL_DEFAULTS: dict[str, dict] = {
    "gru": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0},
    "lstm": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0},
    "tcn": {"hidden_dim": 128, "num_layers": 2, "kernel_size": 3, "dropout": 0.1},
    "transformer": {
        "d_model": 128, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 256, "dropout": 0.1, "pooling": "last",
    },
}


def build_dl_model(model_cfg: ModelConfig, *, input_dim: int, out_dim: int = 1, mode: str = "patient") -> object:
    """Build a DL model from ModelConfig with params dict."""
    name = model_cfg.name
    defaults = _DL_DEFAULTS.get(name, {})
    params = {**defaults, **model_cfg.params}
    is_time = mode == "time"

    if name in ("gru", "lstm"):
        mod = import_module("oneehr.models.recurrent")
        cls_name = "RecurrentTimeModel" if is_time else "RecurrentModel"
        cls = getattr(mod, cls_name)
        return cls(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            num_layers=int(params.get("num_layers", 1)),
            dropout=float(params.get("dropout", 0.0)),
            cell=name,
        )

    if name == "transformer":
        mod = import_module("oneehr.models.transformer")
        cls_name = "TransformerTimeModel" if is_time else "TransformerModel"
        cls = getattr(mod, cls_name)
        kw = dict(
            input_dim=input_dim,
            d_model=int(params.get("d_model", 128)),
            out_dim=out_dim,
            nhead=int(params.get("nhead", 4)),
            num_layers=int(params.get("num_layers", 2)),
            dim_feedforward=int(params.get("dim_feedforward", 256)),
            dropout=float(params.get("dropout", 0.1)),
        )
        if not is_time:
            kw["pooling"] = str(params.get("pooling", "last"))
        return cls(**kw)

    if name == "tcn":
        mod = import_module("oneehr.models.tcn")
        cls_name = "TCNTimeModel" if is_time else "TCNPatientModel"
        cls = getattr(mod, cls_name)
        return cls(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            num_layers=int(params.get("num_layers", 2)),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
        )

    raise ValueError(f"Unsupported DL model: {name!r}")
