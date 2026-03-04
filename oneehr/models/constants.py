"""Shared model-category sets used across CLI and registry."""

from __future__ import annotations

TABULAR_MODELS: frozenset[str] = frozenset(
    {"xgboost", "catboost", "rf", "dt", "gbdt"}
)

DL_MODELS: frozenset[str] = frozenset(
    {
        "gru", "rnn", "lstm", "mlp", "tcn", "transformer",
        "adacare", "stagenet", "retain", "concare", "grasp",
        "mcgru", "dragent",
    }
)

STATIC_ONLY_DL_MODELS: frozenset[str] = frozenset({"mlp"})

STATIC_BRANCH_MODELS: frozenset[str] = frozenset({"dragent", "mcgru", "concare", "grasp"})
