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

    # For tabular models, input_dim is derived from the tabular view at runtime.
    # For DL models, the real input_dim is the number of binned feature columns
    # (computed at runtime); callers should set cfg._dynamic_dim accordingly.
    input_dim = int(getattr(cfg, "_dynamic_dim", 0) or 0)
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

    if name == "stagenet":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.stagenet import StageNetTimeModel

            return BuiltModel(
                model=StageNetTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.stagenet.hidden_dim,
                    conv_size=cfg.model.stagenet.conv_size,
                    levels=cfg.model.stagenet.levels,
                    dropconnect=cfg.model.stagenet.dropconnect,
                    dropout=cfg.model.stagenet.dropout,
                    dropres=cfg.model.stagenet.dropres,
                ),
                kind="dl",
            )
        from oneehr.models.stagenet import StageNetModel

        return BuiltModel(
            model=StageNetModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.stagenet.hidden_dim,
                conv_size=cfg.model.stagenet.conv_size,
                levels=cfg.model.stagenet.levels,
                dropconnect=cfg.model.stagenet.dropconnect,
                dropout=cfg.model.stagenet.dropout,
                dropres=cfg.model.stagenet.dropres,
            ),
            kind="dl",
        )

    if name == "retain":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.retain import RETAINTimeModel

            return BuiltModel(
                model=RETAINTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.retain.hidden_dim,
                    dropout=cfg.model.retain.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.retain import RETAINModel

        return BuiltModel(
            model=RETAINModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.retain.hidden_dim,
                dropout=cfg.model.retain.dropout,
            ),
            kind="dl",
        )

    if name == "adacare":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.adacare import AdaCareTimeModel

            return BuiltModel(
                model=AdaCareTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.adacare.hidden_dim,
                    kernel_size=cfg.model.adacare.kernel_size,
                    kernel_num=cfg.model.adacare.kernel_num,
                    r_v=cfg.model.adacare.r_v,
                    r_c=cfg.model.adacare.r_c,
                    dropout=cfg.model.adacare.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.adacare import AdaCareModel

        return BuiltModel(
            model=AdaCareModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.adacare.hidden_dim,
                kernel_size=cfg.model.adacare.kernel_size,
                kernel_num=cfg.model.adacare.kernel_num,
                r_v=cfg.model.adacare.r_v,
                r_c=cfg.model.adacare.r_c,
                dropout=cfg.model.adacare.dropout,
            ),
            kind="dl",
        )

    if name == "concare":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.concare import ConCareTimeModel

            return BuiltModel(
                model=ConCareTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.concare.hidden_dim,
                    num_heads=cfg.model.concare.num_heads,
                    dropout=cfg.model.concare.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.concare import ConCareModel

        return BuiltModel(
            model=ConCareModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.concare.hidden_dim,
                num_heads=cfg.model.concare.num_heads,
                dropout=cfg.model.concare.dropout,
            ),
            kind="dl",
        )

    if name == "grasp":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.grasp import GRASPTimeModel

            return BuiltModel(
                model=GRASPTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.grasp.hidden_dim,
                    cluster_num=cfg.model.grasp.cluster_num,
                    dropout=cfg.model.grasp.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.grasp import GRASPModel

        return BuiltModel(
            model=GRASPModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.grasp.hidden_dim,
                cluster_num=cfg.model.grasp.cluster_num,
                dropout=cfg.model.grasp.dropout,
            ),
            kind="dl",
        )

    # Static features: for models that implement a dedicated static branch, we
    # pass `static_dim`. For other models, static is expected to be concatenated
    # into the dynamic tensor upstream.
    static_dim = int(getattr(cfg, "_static_dim", 0) or 0)

    if name in {"dragent"}:
        if cfg.task.prediction_mode == "time":
            from oneehr.models.dragent import DrAgentTimeModel

            return BuiltModel(
                model=DrAgentTimeModel(
                    input_dim=input_dim,
                    static_dim=static_dim,
                    hidden_dim=cfg.model.dragent.hidden_dim,
                    dropout=cfg.model.dragent.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.dragent import DrAgentModel

        return BuiltModel(
            model=DrAgentModel(
                input_dim=input_dim,
                static_dim=static_dim,
                hidden_dim=cfg.model.dragent.hidden_dim,
                dropout=cfg.model.dragent.dropout,
            ),
            kind="dl",
        )

    if name in {"mcgru"}:
        if cfg.task.prediction_mode == "time":
            from oneehr.models.mcgru import MCGRUTimeModel

            return BuiltModel(
                model=MCGRUTimeModel(
                    input_dim=input_dim,
                    static_dim=static_dim,
                    hidden_dim=cfg.model.mcgru.hidden_dim,
                    feat_dim=cfg.model.mcgru.hidden_dim // 16 if cfg.model.mcgru.hidden_dim >= 16 else 8,
                    dropout=cfg.model.mcgru.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.mcgru import MCGRUModel

        return BuiltModel(
            model=MCGRUModel(
                input_dim=input_dim,
                static_dim=static_dim,
                hidden_dim=cfg.model.mcgru.hidden_dim,
                feat_dim=cfg.model.mcgru.hidden_dim // 16 if cfg.model.mcgru.hidden_dim >= 16 else 8,
                dropout=cfg.model.mcgru.dropout,
            ),
            kind="dl",
        )

    if name in {"xgboost", "catboost", "rf", "dt", "gbdt"}:
        return BuiltModel(model=None, kind="ml")

    if name == "lstm":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.lstm import LSTMTimeModel

            return BuiltModel(
                model=LSTMTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.lstm.hidden_dim,
                    out_dim=out_dim,
                    num_layers=cfg.model.lstm.num_layers,
                    dropout=cfg.model.lstm.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.lstm import LSTMModel

        return BuiltModel(
            model=LSTMModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.lstm.hidden_dim,
                out_dim=out_dim,
                num_layers=cfg.model.lstm.num_layers,
                dropout=cfg.model.lstm.dropout,
            ),
            kind="dl",
        )

    if name == "mlp":
        if cfg.task.prediction_mode == "time":
            from oneehr.models.mlp import MLPTimeModel

            return BuiltModel(
                model=MLPTimeModel(
                    input_dim=input_dim,
                    hidden_dim=cfg.model.mlp.hidden_dim,
                    num_layers=cfg.model.mlp.num_layers,
                    dropout=cfg.model.mlp.dropout,
                ),
                kind="dl",
            )
        from oneehr.models.mlp import MLPModel

        return BuiltModel(
            model=MLPModel(
                input_dim=input_dim,
                hidden_dim=cfg.model.mlp.hidden_dim,
                num_layers=cfg.model.mlp.num_layers,
                dropout=cfg.model.mlp.dropout,
            ),
            kind="dl",
        )

    raise ValueError(f"Unsupported model.name={name!r}")
