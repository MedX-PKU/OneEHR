"""Model-specific training and inference support.

This module keeps model-zoo registration in ``oneehr.models`` clean while
allowing a small number of models to prepare auxiliary tensors or derived
parameters from preprocessing artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from oneehr.config.schema import ModelConfig, PreprocessConfig


@dataclass(frozen=True)
class PreparedDLArtifacts:
    """Resolved model config plus optional split-aware auxiliary inputs."""

    model_cfg: ModelConfig
    train_extra: dict[str, torch.Tensor] | None = None
    val_extra: dict[str, torch.Tensor] | None = None
    extra_meta: dict[str, object] | None = None


def _updated_model_cfg(model_cfg: ModelConfig, **params: object) -> ModelConfig:
    return type(model_cfg)(name=model_cfg.name, params={**model_cfg.params, **params})


def _load_feature_schema(run_dir: Path) -> list[dict] | None:
    path = run_dir / "preprocess" / "feature_schema.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_obs_mask(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "preprocess" / "obs_mask.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def prepare_dl_artifacts(
    *,
    model_cfg: ModelConfig,
    binned: pd.DataFrame,
    feat_cols: list[str],
    split,
    run_dir: Path,
    preprocess_cfg: PreprocessConfig,
) -> PreparedDLArtifacts:
    """Prepare model-specific params and split-aware extra tensors."""

    model_name = model_cfg.name

    if model_name == "prism":
        from oneehr.models.prism import prepare_prism_training_artifacts

        spec = prepare_prism_training_artifacts(
            model_cfg=model_cfg,
            binned=binned,
            feat_cols=feat_cols,
            split=split,
            run_dir=run_dir,
            preprocess_cfg=preprocess_cfg,
        )
        return PreparedDLArtifacts(**spec)

    if model_name == "safari":
        from oneehr.models.safari import resolve_safari_dim_list

        params = dict(model_cfg.params)
        if "dim_list" not in params:
            feature_schema = _load_feature_schema(run_dir)
            params["dim_list"] = resolve_safari_dim_list(
                feature_schema=feature_schema,
                input_dim=len(feat_cols),
            )
        return PreparedDLArtifacts(model_cfg=_updated_model_cfg(model_cfg, **params))

    if model_name == "pai":
        from oneehr.models.pai import prepare_pai_training_artifacts

        obs_mask = _load_obs_mask(run_dir)
        if obs_mask is None:
            raise ValueError("PAI requires preprocess/obs_mask.parquet")
        spec = prepare_pai_training_artifacts(
            model_cfg=model_cfg,
            binned=binned,
            obs_mask=obs_mask,
            feat_cols=feat_cols,
            split=split,
        )
        return PreparedDLArtifacts(**spec)

    return PreparedDLArtifacts(model_cfg=model_cfg)


def build_inference_extra(
    *,
    model_name: str,
    meta: dict,
    run_dir: Path,
    feat_cols: list[str],
    patient_ids: list[str],
    max_len: int,
) -> dict[str, torch.Tensor]:
    """Build model-specific auxiliary tensors for inference."""

    if model_name == "prism":
        from oneehr.models.prism import prepare_prism_inference_extra

        return prepare_prism_inference_extra(
            meta=meta,
            run_dir=run_dir,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        )

    if model_name == "pai":
        from oneehr.models.pai import build_pai_inference_extra

        obs_mask = _load_obs_mask(run_dir)
        if obs_mask is None:
            raise ValueError("PAI requires preprocess/obs_mask.parquet")
        return build_pai_inference_extra(
            obs_mask=obs_mask,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        )

    return {}
