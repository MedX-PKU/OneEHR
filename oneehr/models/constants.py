"""Shared model-category sets used across CLI and registry."""

from __future__ import annotations

TABULAR_MODELS: frozenset[str] = frozenset({"xgboost", "catboost"})

DL_MODELS: frozenset[str] = frozenset({"gru", "lstm", "tcn", "transformer"})
