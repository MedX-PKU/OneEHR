from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from oneehr.config.schema import (
    CatBoostConfig,
    DTConfig,
    GBDTConfig,
    RFConfig,
    TaskConfig,
    XGBoostConfig,
)


@dataclass
class TabularArtifacts:
    feature_columns: list[str]
    model: object
    kind: str  # xgboost | catboost | rf | dt | gbdt


def train_tabular_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    task: TaskConfig,
    model_cfg: Any,
) -> TabularArtifacts:
    """Train a 2D tabular model.

    Notes:
    - `X_val/y_val` are optional; some models can use them for early stopping.
    - `model_cfg` type depends on `model_name`.
    """

    X_train = X_train.copy()
    feature_columns = list(X_train.columns)

    if model_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cfg: XGBoostConfig = model_cfg

        if task.kind == "binary":
            y_train_u = np.unique(y_train)
            if y_train_u.size < 2:
                raise ValueError(
                    "Binary training split contains a single class; cannot fit XGBoost. "
                    f"classes={y_train_u.tolist()}"
                )
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
            if task.kind == "binary":
                y_val_u = np.unique(y_val)
                if y_val_u.size < 2:
                    model.fit(X_train, y_train)
                    return TabularArtifacts(feature_columns=feature_columns, model=model, kind="xgboost")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="xgboost")

    if model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cfg = model_cfg
        assert isinstance(cfg, CatBoostConfig)

        if task.kind == "binary":
            model = CatBoostClassifier(
                depth=cfg.depth,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                random_state=0,
                verbose=False,
                allow_writing_files=False,
            )
        elif task.kind == "regression":
            model = CatBoostRegressor(
                depth=cfg.depth,
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                random_state=0,
                verbose=False,
                allow_writing_files=False,
            )
        else:
            raise ValueError(f"Unsupported task.kind={task.kind!r}")

        if X_val is not None and y_val is not None and len(X_val) > 0:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        else:
            model.fit(X_train, y_train)

        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="catboost")

    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cfg = model_cfg
        assert isinstance(cfg, RFConfig)

        if task.kind == "binary":
            model = RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                random_state=0,
                n_jobs=-1,
            )
        elif task.kind == "regression":
            model = RandomForestRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                random_state=0,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported task.kind={task.kind!r}")

        model.fit(X_train, y_train)
        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="rf")

    if model_name == "dt":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        cfg = model_cfg
        assert isinstance(cfg, DTConfig)

        if task.kind == "binary":
            model = DecisionTreeClassifier(max_depth=cfg.max_depth, random_state=0)
        elif task.kind == "regression":
            model = DecisionTreeRegressor(max_depth=cfg.max_depth, random_state=0)
        else:
            raise ValueError(f"Unsupported task.kind={task.kind!r}")

        model.fit(X_train, y_train)
        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="dt")

    if model_name == "gbdt":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        cfg = model_cfg
        assert isinstance(cfg, GBDTConfig)

        if task.kind == "binary":
            model = GradientBoostingClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                random_state=0,
            )
        elif task.kind == "regression":
            model = GradientBoostingRegressor(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                random_state=0,
            )
        else:
            raise ValueError(f"Unsupported task.kind={task.kind!r}")

        model.fit(X_train, y_train)
        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="gbdt")

    raise ValueError(f"Unsupported tabular model_name={model_name!r}")


def predict_tabular(art: TabularArtifacts, X: pd.DataFrame, task: TaskConfig) -> np.ndarray:
    X = X[art.feature_columns]
    if task.kind == "binary":
        return art.model.predict_proba(X)[:, 1]
    if task.kind == "regression":
        return art.model.predict(X)
    raise ValueError(f"Unsupported task.kind={task.kind!r}")


def predict_tabular_logits(art: TabularArtifacts, X: pd.DataFrame, task: TaskConfig) -> np.ndarray | None:
    """Return decision values (log-odds) when available.

    Used for post-hoc calibration (temperature/platt) on logits.
    """

    if art.kind != "xgboost":
        return None
    if task.kind != "binary":
        return None
    X = X[art.feature_columns]
    return art.model.predict(X, output_margin=True)

