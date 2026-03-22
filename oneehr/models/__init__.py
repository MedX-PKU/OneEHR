"""Model architectures and registry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from oneehr.config.schema import ModelConfig, TaskConfig

TABULAR_MODELS: frozenset[str] = frozenset({"xgboost", "catboost", "rf", "dt", "gbdt", "lr"})
DL_MODELS: frozenset[str] = frozenset({
    "gru", "lstm", "rnn", "tcn", "transformer",
    "mlp", "cnn", "grud", "sand", "dipole",
    "adacare", "stagenet", "retain",
    "concare", "grasp", "mcgru", "dragent",
    "deepr", "mamba", "jamba", "prism",
    "m3care", "safari", "pai",
    "deepsurv", "deephit",
})


@dataclass(frozen=True)
class BuiltModel:
    model: object
    kind: str  # dl | ml


# Default hyperparams per DL model type
_DL_DEFAULTS: dict[str, dict] = {
    "gru": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0},
    "lstm": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0},
    "rnn": {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0},
    "tcn": {"hidden_dim": 128, "num_layers": 2, "kernel_size": 3, "dropout": 0.1},
    "cnn": {"hidden_dim": 128, "num_layers": 2, "kernel_size": 3, "dropout": 0.1},
    "transformer": {
        "d_model": 128, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 256, "dropout": 0.1, "pooling": "last",
    },
    "grud": {"hidden_dim": 128, "dropout": 0.0},
    "sand": {
        "d_model": 128, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 256, "kernel_size": 3, "interp_points": 8, "dropout": 0.1,
    },
    "dipole": {"hidden_dim": 128, "attention_type": "general", "dropout": 0.1},
    "mlp": {"hidden_dim": 128, "dropout": 0.0},
    "adacare": {"hidden_dim": 128, "kernel_size": 2, "kernel_num": 64, "dropout": 0.5},
    "stagenet": {"chunk_size": 128, "levels": 3, "conv_size": 10, "dropout": 0.3},
    "retain": {"hidden_dim": 128, "dropout": 0.5},
    "concare": {"hidden_dim": 128, "num_heads": 4, "dropout": 0.5},
    "grasp": {"hidden_dim": 128, "cluster_num": 12, "dropout": 0.5},
    "mcgru": {"hidden_dim": 32, "feat_dim": 8, "dropout": 0.0},
    "dragent": {"hidden_dim": 128, "n_actions": 10, "n_units": 64, "dropout": 0.5, "lamda": 0.5},
    "deepr": {"hidden_dim": 128, "window": 1, "dropout": 0.0},
    "mamba": {"hidden_dim": 128, "num_layers": 2, "state_size": 16, "conv_kernel": 4, "dropout": 0.1},
    "jamba": {
        "hidden_dim": 128, "num_transformer_layers": 2, "num_mamba_layers": 6,
        "heads": 4, "state_size": 16, "conv_kernel": 4, "dropout": 0.3,
    },
    "prism": {
        "hidden_dim": 32, "feat_dim": 8, "n_clusters": 10, "calib": True, "dropout": 0.0,
    },
    "m3care": {
        "hidden_dim": 128, "num_heads": 4, "dim_feedforward": 256, "dropout": 0.1, "num_layers": 1,
    },
    "safari": {
        "hidden_dim": 32, "n_clu": 8, "dropout": 0.5,
    },
    "pai": {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.0, "prompt_init": "median",
    },
    "deepsurv": {
        "hidden_dim": 128, "num_layers": 2, "dropout": 0.1,
    },
    "deephit": {
        "hidden_dim": 128, "num_time_bins": 10, "num_layers": 2, "dropout": 0.1,
    },
}


def build_dl_model(model_cfg: ModelConfig, *, input_dim: int, out_dim: int = 1, mode: str = "patient") -> object:
    """Build a DL model from ModelConfig with params dict."""
    name = model_cfg.name
    defaults = _DL_DEFAULTS.get(name, {})
    params = {**defaults, **model_cfg.params}
    is_time = mode == "time"

    if name in ("gru", "lstm", "rnn"):
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

    if name == "cnn":
        mod = import_module("oneehr.models.cnn")
        cls_name = "CNNTimeModel" if is_time else "CNNPatientModel"
        cls = getattr(mod, cls_name)
        return cls(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            num_layers=int(params.get("num_layers", 2)),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
        )

    if name == "mlp":
        mod = import_module("oneehr.models.mlp")
        cls_name = "MLPTimeModel" if is_time else "MLPModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            dropout=float(params.get("dropout", 0.0)),
        )

    if name == "grud":
        mod = import_module("oneehr.models.grud")
        cls_name = "GRUDTimeModel" if is_time else "GRUDModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            dropout=float(params.get("dropout", 0.0)),
            feature_means=params.get("feature_means"),
        )

    if name == "sand":
        mod = import_module("oneehr.models.sand")
        cls_name = "SAnDTimeModel" if is_time else "SAnDModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            d_model=int(params.get("d_model", 128)),
            out_dim=out_dim,
            nhead=int(params.get("nhead", 4)),
            num_layers=int(params.get("num_layers", 2)),
            dim_feedforward=int(params.get("dim_feedforward", 256)),
            kernel_size=int(params.get("kernel_size", 3)),
            interp_points=int(params.get("interp_points", 8)),
            dropout=float(params.get("dropout", 0.1)),
        )

    if name == "dipole":
        mod = import_module("oneehr.models.dipole")
        cls_name = "DipoleTimeModel" if is_time else "DipoleModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            attention_type=str(params.get("attention_type", "general")),
            dropout=float(params.get("dropout", 0.1)),
        )

    if name == "adacare":
        mod = import_module("oneehr.models.adacare")
        cls_name = "AdaCareTimeModel" if is_time else "AdaCareModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            kernel_size=int(params.get("kernel_size", 2)),
            kernel_num=int(params.get("kernel_num", 64)),
            dropout=float(params.get("dropout", 0.5)),
        )

    if name == "stagenet":
        mod = import_module("oneehr.models.stagenet")
        cls_name = "StageNetTimeModel" if is_time else "StageNetModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            chunk_size=int(params.get("chunk_size", 128)),
            levels=int(params.get("levels", 3)),
            conv_size=int(params.get("conv_size", 10)),
            out_dim=out_dim,
            dropout=float(params.get("dropout", 0.3)),
        )

    if name == "retain":
        mod = import_module("oneehr.models.retain")
        cls_name = "RETAINTimeModel" if is_time else "RETAINModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            dropout=float(params.get("dropout", 0.5)),
        )

    if name == "concare":
        mod = import_module("oneehr.models.concare")
        cls_name = "ConCareTimeModel" if is_time else "ConCareModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            num_heads=int(params.get("num_heads", 4)),
            out_dim=out_dim,
            static_dim=int(params.get("static_dim", 0)),
            dropout=float(params.get("dropout", 0.5)),
        )

    if name == "grasp":
        mod = import_module("oneehr.models.grasp")
        cls_name = "GRASPTimeModel" if is_time else "GRASPModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            cluster_num=int(params.get("cluster_num", 12)),
            out_dim=out_dim,
            static_dim=int(params.get("static_dim", 0)),
            dropout=float(params.get("dropout", 0.5)),
        )

    if name == "mcgru":
        mod = import_module("oneehr.models.mcgru")
        cls_name = "MCGRUTimeModel" if is_time else "MCGRUModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 32)),
            feat_dim=int(params.get("feat_dim", 8)),
            out_dim=out_dim,
            static_dim=int(params.get("static_dim", 0)),
            dropout=float(params.get("dropout", 0.0)),
        )

    if name == "dragent":
        mod = import_module("oneehr.models.dragent")
        cls_name = "DrAgentTimeModel" if is_time else "DrAgentModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            static_dim=int(params.get("static_dim", 0)),
            n_actions=int(params.get("n_actions", 10)),
            n_units=int(params.get("n_units", 64)),
            dropout=float(params.get("dropout", 0.5)),
            lamda=float(params.get("lamda", 0.5)),
        )

    if name == "deepr":
        mod = import_module("oneehr.models.deepr")
        cls_name = "DeeprTimeModel" if is_time else "DeeprModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            window=int(params.get("window", 1)),
            dropout=float(params.get("dropout", 0.0)),
        )

    if name == "mamba":
        mod = import_module("oneehr.models.mamba")
        cls_name = "EHRMambaTimeModel" if is_time else "EHRMambaModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            num_layers=int(params.get("num_layers", 2)),
            state_size=int(params.get("state_size", 16)),
            conv_kernel=int(params.get("conv_kernel", 4)),
            dropout=float(params.get("dropout", 0.1)),
        )

    if name == "jamba":
        mod = import_module("oneehr.models.jamba")
        cls_name = "JambaTimeModel" if is_time else "JambaModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            num_transformer_layers=int(params.get("num_transformer_layers", 2)),
            num_mamba_layers=int(params.get("num_mamba_layers", 6)),
            heads=int(params.get("heads", 4)),
            state_size=int(params.get("state_size", 16)),
            conv_kernel=int(params.get("conv_kernel", 4)),
            dropout=float(params.get("dropout", 0.3)),
        )

    if name == "prism":
        mod = import_module("oneehr.models.prism")
        cls_name = "PRISMTimeModel" if is_time else "PRISMModel"
        cls = getattr(mod, cls_name)
        return cls(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 32)),
            feat_dim=int(params.get("feat_dim", 8)),
            out_dim=out_dim,
            static_dim=int(params.get("static_dim", 0)),
            dim_list=params.get("dim_list"),
            centers=params.get("centers"),
            n_clusters=int(params.get("n_clusters", 10)),
            calib=bool(params.get("calib", True)),
            dropout=float(params.get("dropout", 0.0)),
        )

    if name == "m3care":
        mod = import_module("oneehr.models.m3care")
        cls_name = "M3CareTimeModel" if is_time else "M3CareModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            num_heads=int(params.get("num_heads", 4)),
            dim_feedforward=int(params.get("dim_feedforward", 256)),
            dropout=float(params.get("dropout", 0.1)),
            num_layers=int(params.get("num_layers", 1)),
        )

    if name == "safari":
        mod = import_module("oneehr.models.safari")
        cls_name = "SafariTimeModel" if is_time else "SafariModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 32)),
            out_dim=out_dim,
            dim_list=params.get("dim_list"),
            n_clu=int(params.get("n_clu", 8)),
            static_dim=int(params.get("static_dim", 0)),
            dropout=float(params.get("dropout", 0.5)),
        )

    if name == "pai":
        mod = import_module("oneehr.models.pai")
        cls_name = "PAITimeModel" if is_time else "PAIModel"
        return getattr(mod, cls_name)(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            out_dim=out_dim,
            num_layers=int(params.get("num_layers", 1)),
            dropout=float(params.get("dropout", 0.0)),
            prompt_init_values=params.get("prompt_init_values"),
        )

    if name == "deepsurv":
        mod = import_module("oneehr.models.survival")
        return mod.DeepSurv(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            out_dim=out_dim,
        )

    if name == "deephit":
        mod = import_module("oneehr.models.survival")
        return mod.DeepHit(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            num_time_bins=int(params.get("num_time_bins", 10)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            out_dim=out_dim,
        )

    raise ValueError(f"Unsupported DL model: {name!r}")
