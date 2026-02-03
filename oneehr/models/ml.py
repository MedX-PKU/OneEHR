from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneehr.config.schema import (
    CatBoostConfig,
    DTConfig,
    GBDTConfig,
    RFConfig,
    TaskConfig,
)


@dataclass
class MLArtifacts:
    feature_columns: list[str]
    model: object


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    task: TaskConfig,
    cfg: CatBoostConfig,
) -> MLArtifacts:
    from catboost import CatBoostClassifier, CatBoostRegressor

    X_train = X_train.copy()
    feature_columns = list(X_train.columns)

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

    return MLArtifacts(feature_columns=feature_columns, model=model)


def train_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    task: TaskConfig,
    cfg: RFConfig,
) -> MLArtifacts:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X_train = X_train.copy()
    feature_columns = list(X_train.columns)

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
    return MLArtifacts(feature_columns=feature_columns, model=model)


def train_dt(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    task: TaskConfig,
    cfg: DTConfig,
) -> MLArtifacts:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    X_train = X_train.copy()
    feature_columns = list(X_train.columns)

    if task.kind == "binary":
        model = DecisionTreeClassifier(max_depth=cfg.max_depth, random_state=0)
    elif task.kind == "regression":
        model = DecisionTreeRegressor(max_depth=cfg.max_depth, random_state=0)
    else:
        raise ValueError(f"Unsupported task.kind={task.kind!r}")

    model.fit(X_train, y_train)
    return MLArtifacts(feature_columns=feature_columns, model=model)


def train_gbdt(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    task: TaskConfig,
    cfg: GBDTConfig,
) -> MLArtifacts:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    X_train = X_train.copy()
    feature_columns = list(X_train.columns)

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
    return MLArtifacts(feature_columns=feature_columns, model=model)


def predict_ml(art: MLArtifacts, X: pd.DataFrame, task: TaskConfig) -> np.ndarray:
    X = X[art.feature_columns]
    if task.kind == "binary":
        return art.model.predict_proba(X)[:, 1]
    if task.kind == "regression":
        return art.model.predict(X)
    raise ValueError(f"Unsupported task.kind={task.kind!r}")

