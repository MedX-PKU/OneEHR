"""Model-specific training and inference support.

This module keeps model-zoo registration in ``oneehr.models`` clean while
allowing a small number of models to prepare auxiliary tensors or derived
parameters from preprocessing artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from oneehr.config.schema import ModelConfig, PreprocessConfig
from oneehr.models.adapters import (
    build_group_mask_tensor,
    build_group_sequence_tensor,
    build_missing_mask_tensor,
    build_time_delta_map,
    build_time_delta_tensor,
    build_visit_time_map,
    build_visit_time_tensor,
    load_feature_schema,
    load_obs_mask,
    resolve_feature_groups,
)
from oneehr.models.kg import build_lightweight_kg


@dataclass(frozen=True)
class PreparedDLArtifacts:
    """Resolved model config plus optional split-aware auxiliary inputs."""

    model_cfg: ModelConfig
    train_extra: dict[str, torch.Tensor] | None = None
    val_extra: dict[str, torch.Tensor] | None = None
    extra_meta: dict[str, object] | None = None


def _updated_model_cfg(model_cfg: ModelConfig, **params: object) -> ModelConfig:
    return type(model_cfg)(name=model_cfg.name, params={**model_cfg.params, **params})


def _compute_observed_feature_means(
    *,
    binned: pd.DataFrame,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: set[str],
) -> torch.Tensor:
    feat = binned[binned["patient_id"].astype(str).isin(patient_ids)].copy()
    mask = obs_mask[obs_mask["patient_id"].astype(str).isin(patient_ids)].copy()
    merged = feat[["patient_id", "bin_time", *feat_cols]].merge(
        mask[["patient_id", "bin_time", *feat_cols]],
        on=["patient_id", "bin_time"],
        how="inner",
        suffixes=("", "__obs"),
    )
    out = []
    for col in feat_cols:
        obs_col = f"{col}__obs"
        valid = merged[obs_col] > 0.5
        out.append(float(merged.loc[valid, col].mean()) if valid.any() else 0.0)
    return torch.tensor(out, dtype=torch.float32)


def _sequence_extra_for_patient_ids(
    *,
    binned: pd.DataFrame,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: list[str],
    bin_size: str,
) -> dict[str, torch.Tensor]:
    patient_id_set = set(str(pid) for pid in patient_ids)
    subset_binned = binned[binned["patient_id"].astype(str).isin(patient_id_set)].copy()
    subset_obs = obs_mask[obs_mask["patient_id"].astype(str).isin(patient_id_set)].copy()
    visit_time_map = build_visit_time_map(
        binned=subset_binned,
        patient_ids=patient_id_set,
        bin_size=bin_size,
    )
    time_delta_map = build_time_delta_map(
        obs_mask=subset_obs,
        feat_cols=feat_cols,
        patient_ids=patient_id_set,
        bin_size=bin_size,
    )
    max_len = max((arr.shape[0] for arr in visit_time_map.values()), default=0)
    return {
        "missing_mask": build_missing_mask_tensor(
            obs_mask=subset_obs,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        ),
        "time_delta": build_time_delta_tensor(
            time_delta_map,
            patient_ids=patient_ids,
            max_len=max_len,
            input_dim=len(feat_cols),
        ),
        "visit_time": build_visit_time_tensor(
            visit_time_map,
            patient_ids=patient_ids,
            max_len=max_len,
        ),
    }


def _build_inference_sequence_extra(
    *,
    run_dir: Path,
    feat_cols: list[str],
    patient_ids: list[str],
    max_len: int,
) -> dict[str, torch.Tensor]:
    from oneehr.artifacts.manifest import read_manifest

    manifest = read_manifest(run_dir)
    bin_size = str(manifest.get("config", {}).get("preprocess", {}).get("bin_size", "1d"))
    obs_mask = load_obs_mask(run_dir)
    if obs_mask is None:
        raise ValueError("Model requires preprocess/obs_mask.parquet")
    binned = pd.read_parquet(run_dir / "preprocess" / "binned.parquet")
    patient_id_set = set(str(pid) for pid in patient_ids)
    subset_obs = obs_mask[obs_mask["patient_id"].astype(str).isin(patient_id_set)].copy()
    subset_binned = binned[binned["patient_id"].astype(str).isin(patient_id_set)].copy()
    visit_time_map = build_visit_time_map(
        binned=subset_binned,
        patient_ids=patient_id_set,
        bin_size=bin_size,
    )
    time_delta_map = build_time_delta_map(
        obs_mask=subset_obs,
        feat_cols=feat_cols,
        patient_ids=patient_id_set,
        bin_size=bin_size,
    )
    return {
        "missing_mask": build_missing_mask_tensor(
            obs_mask=subset_obs,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        ),
        "time_delta": build_time_delta_tensor(
            time_delta_map,
            patient_ids=patient_ids,
            max_len=max_len,
            input_dim=len(feat_cols),
        ),
        "visit_time": build_visit_time_tensor(
            visit_time_map,
            patient_ids=patient_ids,
            max_len=max_len,
        ),
    }


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
            feature_schema = load_feature_schema(run_dir)
            params["dim_list"] = resolve_safari_dim_list(
                feature_schema=feature_schema,
                input_dim=len(feat_cols),
            )
        return PreparedDLArtifacts(model_cfg=_updated_model_cfg(model_cfg, **params))

    if model_name == "pai":
        from oneehr.models.pai import prepare_pai_training_artifacts

        obs_mask = load_obs_mask(run_dir)
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

    if model_name == "grud":
        obs_mask = load_obs_mask(run_dir)
        if obs_mask is None:
            raise ValueError("GRU-D requires preprocess/obs_mask.parquet")
        params = dict(model_cfg.params)
        if "feature_means" not in params:
            params["feature_means"] = _compute_observed_feature_means(
                binned=binned,
                obs_mask=obs_mask,
                feat_cols=feat_cols,
                patient_ids=set(str(pid) for pid in split.train),
            )
        train_extra = _sequence_extra_for_patient_ids(
            binned=binned,
            obs_mask=obs_mask,
            feat_cols=feat_cols,
            patient_ids=[str(pid) for pid in split.train],
            bin_size=preprocess_cfg.bin_size,
        )
        val_extra = _sequence_extra_for_patient_ids(
            binned=binned,
            obs_mask=obs_mask,
            feat_cols=feat_cols,
            patient_ids=[str(pid) for pid in split.val],
            bin_size=preprocess_cfg.bin_size,
        )
        return PreparedDLArtifacts(
            model_cfg=_updated_model_cfg(model_cfg, **params),
            train_extra=train_extra,
            val_extra=val_extra,
        )

    if model_name in {"lsan", "hitanet"}:
        feature_schema = load_feature_schema(run_dir)
        groups = resolve_feature_groups(feat_cols=feat_cols, feature_schema=feature_schema)
        params = {
            "group_indices": [group.indices for group in groups],
            "group_names": [group.name for group in groups],
        }
        if model_name == "hitanet":
            obs_mask = load_obs_mask(run_dir)
            if obs_mask is None:
                raise ValueError("hitanet requires preprocess/obs_mask.parquet")
            return PreparedDLArtifacts(
                model_cfg=_updated_model_cfg(model_cfg, **params),
                train_extra=_sequence_extra_for_patient_ids(
                    binned=binned,
                    obs_mask=obs_mask,
                    feat_cols=feat_cols,
                    patient_ids=[str(pid) for pid in split.train],
                    bin_size=preprocess_cfg.bin_size,
                ),
                val_extra=_sequence_extra_for_patient_ids(
                    binned=binned,
                    obs_mask=obs_mask,
                    feat_cols=feat_cols,
                    patient_ids=[str(pid) for pid in split.val],
                    bin_size=preprocess_cfg.bin_size,
                ),
            )
        return PreparedDLArtifacts(
            model_cfg=_updated_model_cfg(model_cfg, **params)
        )

    time_extra_models = {"mtand", "raindrop", "contiformer", "teco"}
    if model_name in time_extra_models:
        obs_mask = load_obs_mask(run_dir)
        if obs_mask is None:
            raise ValueError(f"{model_name} requires preprocess/obs_mask.parquet")
        return PreparedDLArtifacts(
            model_cfg=model_cfg,
            train_extra=_sequence_extra_for_patient_ids(
                binned=binned,
                obs_mask=obs_mask,
                feat_cols=feat_cols,
                patient_ids=[str(pid) for pid in split.train],
                bin_size=preprocess_cfg.bin_size,
            ),
            val_extra=_sequence_extra_for_patient_ids(
                binned=binned,
                obs_mask=obs_mask,
                feat_cols=feat_cols,
                patient_ids=[str(pid) for pid in split.val],
                bin_size=preprocess_cfg.bin_size,
            ),
        )

    if model_name in {"graphcare", "kerprint", "protoehr"}:
        obs_mask = load_obs_mask(run_dir)
        if obs_mask is None:
            raise ValueError(f"{model_name} requires preprocess/obs_mask.parquet")
        feature_schema = load_feature_schema(run_dir)
        kg = build_lightweight_kg(
            binned=binned,
            obs_mask=obs_mask,
            feat_cols=feat_cols,
            feature_schema=feature_schema,
            split=split,
            bin_size=preprocess_cfg.bin_size,
            kg_source=str(model_cfg.params.get("kg_source", "lightweight")),
            external_kg_path=model_cfg.params.get("external_kg_path"),
            kg_top_k=int(model_cfg.params.get("kg_top_k", 6)),
            kg_min_cooccurrence=int(model_cfg.params.get("kg_min_cooccurrence", 2)),
            kg_ontology=str(model_cfg.params.get("kg_ontology", "auto")),
        )
        return PreparedDLArtifacts(
            model_cfg=_updated_model_cfg(
                model_cfg,
                group_indices=[group.indices for group in kg.groups],
                group_names=kg.group_names,
                global_adj=kg.global_adj,
            ),
            train_extra=kg.train_extra,
            val_extra=kg.val_extra,
            extra_meta=kg.extra_meta,
        )

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

        obs_mask = load_obs_mask(run_dir)
        if obs_mask is None:
            raise ValueError("PAI requires preprocess/obs_mask.parquet")
        return build_pai_inference_extra(
            obs_mask=obs_mask,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        )

    if model_name == "grud":
        return _build_inference_sequence_extra(
            run_dir=run_dir,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        )

    if model_name in {"hitanet", "mtand", "raindrop", "contiformer", "teco"}:
        return _build_inference_sequence_extra(
            run_dir=run_dir,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        )

    if model_name in {"graphcare", "kerprint", "protoehr"}:
        from oneehr.artifacts.manifest import read_manifest

        manifest = read_manifest(run_dir)
        bin_size = manifest.get("config", {}).get("preprocess", {}).get("bin_size", "1d")
        binned = pd.read_parquet(run_dir / "preprocess" / "binned.parquet")
        obs_mask = load_obs_mask(run_dir)
        if obs_mask is None:
            raise ValueError(f"{model_name} requires preprocess/obs_mask.parquet")
        test_ids = [str(pid) for pid in patient_ids]
        groups = resolve_feature_groups(
            feat_cols=feat_cols,
            feature_schema=load_feature_schema(run_dir),
        )
        subset_binned = binned[binned["patient_id"].astype(str).isin(set(test_ids))].copy()
        subset_obs = obs_mask[obs_mask["patient_id"].astype(str).isin(set(test_ids))].copy()
        group_values = build_group_sequence_tensor(
            binned=subset_binned,
            groups=groups,
            feat_cols=feat_cols,
            patient_ids=test_ids,
            max_len=max_len,
            reduce="mean",
        )
        group_mask = build_group_mask_tensor(
            obs_mask=subset_obs,
            groups=groups,
            feat_cols=feat_cols,
            patient_ids=test_ids,
            max_len=max_len,
        )
        subset_binned = binned[binned["patient_id"].astype(str).isin(set(test_ids))].copy()
        visit_time_map = build_visit_time_map(
            binned=subset_binned,
            patient_ids=set(test_ids),
            bin_size=bin_size,
        )
        return {
            "group_values": group_values,
            "group_mask": group_mask,
            "visit_time": build_visit_time_tensor(visit_time_map, patient_ids=test_ids, max_len=max_len),
        }

    return {}
