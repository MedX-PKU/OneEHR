from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneehr.config.schema import TaskConfig, XGBoostConfig


@dataclass
class XGBArtifacts:
    feature_columns: list[str]
    model: object


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    task: TaskConfig,
    cfg: XGBoostConfig,
) -> XGBArtifacts:
    from xgboost import XGBClassifier, XGBRegressor

    X_train = X_train.copy()
    feature_columns = list(X_train.columns)

    if task.kind == "binary":
        model = XGBClassifier(
            max_depth=cfg.max_depth,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            min_child_weight=cfg.min_child_weight,
            eval_metric="logloss",
        )
    elif task.kind == "regression":
        model = XGBRegressor(
            max_depth=cfg.max_depth,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            min_child_weight=cfg.min_child_weight,
        )
    else:
        raise ValueError(f"Unsupported task.kind={task.kind!r}")

    if X_val is not None and y_val is not None and len(X_val) > 0:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return XGBArtifacts(feature_columns=feature_columns, model=model)


def predict_xgboost(art: XGBArtifacts, X: pd.DataFrame, task: TaskConfig) -> np.ndarray:
    X = X[art.feature_columns]
    if task.kind == "binary":
        return art.model.predict_proba(X)[:, 1]
    if task.kind == "regression":
        return art.model.predict(X)
    raise ValueError(f"Unsupported task.kind={task.kind!r}")

