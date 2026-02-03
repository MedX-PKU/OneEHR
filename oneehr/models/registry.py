from __future__ import annotations

from dataclasses import dataclass

from oneehr.config.schema import ExperimentConfig


@dataclass(frozen=True)
class BuiltModel:
    model: object
    kind: str  # dl | ml


def build_model(cfg: ExperimentConfig) -> BuiltModel:
    """Build a model instance from config.

    This function centralizes the name->implementation mapping so the CLI and
    other entrypoints don't grow large if/elif ladders.
    """

    name = cfg.model.name

    input_dim = int(cfg.preprocess.top_k_codes or 0)
    out_dim = 1

    if name == "gru":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.gru import GRUTimeModel

            return BuiltModel(
                model=GRUTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.gru.hidden_dim,
                    out_dim=out_dim,
                    num_layers=cfg.model.gru.num_layers,
                    dropout=cfg.model.gru.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.gru import GRUModel

        return BuiltModel(
            model=GRUModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.gru.hidden_dim,
                out_dim=out_dim,
                num_layers=cfg.model.gru.num_layers,
                dropout=cfg.model.gru.dropout,
            ),
            kind="dl",
        )

    if name == "rnn":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.rnn import RNNTimeModel

            return BuiltModel(
                model=RNNTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.rnn.hidden_dim,
                    out_dim=out_dim,
                    num_layers=cfg.model.rnn.num_layers,
                    dropout=cfg.model.rnn.dropout,
                    bidirectional=cfg.model.rnn.bidirectional,
                    nonlinearity=cfg.model.rnn.nonlinearity,
                ),
                kind="dl",
            )
        from oneehr.models.rnn import RNNModel

        return BuiltModel(
            model=RNNModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.rnn.hidden_dim,
                out_dim=out_dim,
                num_layers=cfg.model.rnn.num_layers,
                dropout=cfg.model.rnn.dropout,
                bidirectional=cfg.model.rnn.bidirectional,
                nonlinearity=cfg.model.rnn.nonlinearity,
            ),
            kind="dl",
        )

    if name == "transformer":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.transformer import TransformerTimeModel

            return BuiltModel(
                model=TransformerTimeModel(
                    input_dim=input_dim,
                    d_model=cfg.model.transformer.d_model,
                    out_dim=out_dim,
                    nhead=cfg.model.transformer.nhead,
                    num_layers=cfg.model.transformer.num_layers,
                    dim_feedforward=cfg.model.transformer.dim_feedforward,
                    dropout=cfg.model.transformer.dropout,
                    pooling=cfg.model.transformer.pooling,
                ),
                kind="dl",
            )
        from oneehr.models.transformer import TransformerModel

        return BuiltModel(
            model=TransformerModel(
                input_dim=input_dim,
                d_model=cfg.model.transformer.d_model,
                out_dim=out_dim,
                nhead=cfg.model.transformer.nhead,
                num_layers=cfg.model.transformer.num_layers,
                dim_feedforward=cfg.model.transformer.dim_feedforward,
                dropout=cfg.model.transformer.dropout,
                pooling=cfg.model.transformer.pooling,
            ),
            kind="dl",
        )

    if name == "tcn":
        if cfg.task.prediction_mode != "time":
            raise ValueError("model.name='tcn' currently supports prediction_mode='time' only")
        from oneehr.models.tcn import TCNTimeModel

        return BuiltModel(
            model=TCNTimeModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.tcn.hidden_dim,
                out_dim=out_dim,
                num_layers=cfg.model.tcn.num_layers,
                kernel_size=cfg.model.tcn.kernel_size,
                dropout=cfg.model.tcn.dropout,
            ),
            kind="dl",
        )

    if name in {"dr_agent", "dr-agent"}:
        static_dim = 0 if cfg.static_features is None or not cfg.static_features.enabled else len(cfg.static_features.cols)
        if cfg.task.prediction_mode == "time":
            from oneehr.models.dr_agent import DrAgentTimeModel

            return BuiltModel(
                model=DrAgentTimeModel(
                    input_dim=input_dim,
                    static_dim=static_dim,
                    hidden_dim=cfg.model.agent.hidden_dim,
                    dropout=cfg.model.agent.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.dr_agent import DrAgentModel

        return BuiltModel(
            model=DrAgentModel(
                input_dim=input_dim,
                static_dim=static_dim,
                hidden_dim=cfg.model.agent.hidden_dim,
                dropout=cfg.model.agent.dropout,
            ),
            kind="dl",
        )

    if name in {"xgboost", "catboost", "rf", "dt", "gbdt"}:
        return BuiltModel(model=None, kind="ml")

    raise ValueError(f"Unsupported model.name={name!r}")
