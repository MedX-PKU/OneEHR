from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oneehr.config.schema import TaskConfig


@dataclass
class TabularArtifacts:
    feature_columns: list[str]
    model: object
    kind: str  # xgboost | catboost


def save_tabular_model(art: TabularArtifacts, model_dir: str | Path) -> None:
    import json

    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "feature_columns.json").write_text(json.dumps(art.feature_columns), encoding="utf-8")

    if art.kind == "xgboost":
        art.model.save_model(d / "model.json")
        return
    if art.kind == "catboost":
        art.model.save_model(d / "model.cbm")
        return
    raise ValueError(f"Unsupported tabular kind={art.kind!r}")


def load_tabular_model(model_dir: str | Path, *, task: TaskConfig, kind: str) -> TabularArtifacts:
    import json

    d = Path(model_dir)
    feature_columns = json.loads((d / "feature_columns.json").read_text(encoding="utf-8"))

    if kind == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        model = XGBClassifier() if task.kind == "binary" else XGBRegressor()
        model.load_model(d / "model.json")
        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="xgboost")

    if kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        model = CatBoostClassifier() if task.kind == "binary" else CatBoostRegressor()
        model.load_model(d / "model.cbm")
        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="catboost")

    raise ValueError(f"Unsupported tabular kind={kind!r}")


def train_tabular_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    task: TaskConfig,
    params: dict,
    seed: int = 0,
) -> TabularArtifacts:
    """Train a 2D tabular model (XGBoost or CatBoost) using params dict."""

    X_train = X_train.copy()
    feature_columns = list(X_train.columns)

    if model_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        # Defaults merged with user params
        defaults = dict(
            max_depth=6, n_estimators=500, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=1.0, random_state=seed,
        )
        kw = {**defaults, **params, "random_state": seed}

        if task.kind == "binary":
            kw["eval_metric"] = "logloss"
            model = XGBClassifier(**kw)
        elif task.kind == "regression":
            model = XGBRegressor(**kw)
        else:
            raise ValueError(f"Unsupported task.kind={task.kind!r}")

        if X_val is not None and y_val is not None and len(X_val) > 0:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="xgboost")

    if model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        defaults = dict(depth=6, n_estimators=500, learning_rate=0.05)
        kw = {**defaults, **params, "random_state": seed, "verbose": False, "allow_writing_files": False}

        if task.kind == "binary":
            model = CatBoostClassifier(**kw)
        elif task.kind == "regression":
            model = CatBoostRegressor(**kw)
        else:
            raise ValueError(f"Unsupported task.kind={task.kind!r}")

        if X_val is not None and y_val is not None and len(X_val) > 0:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        else:
            model.fit(X_train, y_train)

        return TabularArtifacts(feature_columns=feature_columns, model=model, kind="catboost")

    raise ValueError(f"Unsupported tabular model_name={model_name!r}")


def predict_tabular(art: TabularArtifacts, X: pd.DataFrame, task: TaskConfig) -> np.ndarray:
    X = X[art.feature_columns]
    if task.kind == "binary":
        return art.model.predict_proba(X)[:, 1]
    if task.kind == "regression":
        return art.model.predict(X)
    raise ValueError(f"Unsupported task.kind={task.kind!r}")
