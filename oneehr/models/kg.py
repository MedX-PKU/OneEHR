"""Lightweight knowledge-graph helpers for graph-enhanced EHR models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oneehr.medcode.atc import ATCHierarchy
from oneehr.medcode.icd import ICD9, ICD10
from oneehr.models.adapters import (
    FeatureGroup,
    build_group_mask_tensor,
    build_group_sequence_tensor,
    build_visit_time_map,
    build_visit_time_tensor,
    normalize_feature_name,
    resolve_feature_groups,
)
from oneehr.models.graph import normalize_adjacency


@dataclass(frozen=True)
class KGArtifacts:
    groups: list[FeatureGroup]
    group_names: list[str]
    global_adj: torch.Tensor
    train_extra: dict[str, torch.Tensor]
    val_extra: dict[str, torch.Tensor]
    extra_meta: dict[str, object]


def _base_code(name: str) -> str:
    raw = normalize_feature_name(name)
    if "__" in raw:
        raw = raw.split("__", 1)[0]
    return raw


def _ontology_bucket(name: str, ontology: str) -> str | None:
    raw = _base_code(name).upper()
    if raw.startswith("DX_"):
        code = raw[3:]
        if not code:
            return None
        if ontology in {"auto", "icd"}:
            if code[:1].isalpha():
                return f"ICD10::{ICD10.category(code)}::{ICD10.chapter(code)}"
            return f"ICD9::{ICD9.category(code)}::{ICD9.chapter(code)}"
    if raw.startswith("RX_") and ontology in {"auto", "atc"}:
        code = raw[3:]
        if not code:
            return None
        atc = ATCHierarchy()
        return f"ATC::{atc.group(code, level=1)}::{atc.group_name(code, level=1) or 'unknown'}"
    if ontology == "none":
        return None
    if "_" in raw:
        return f"LEX::{raw.split('_', 1)[0]}"
    return None


def _load_external_graph(path: Path, group_names: list[str]) -> torch.Tensor:
    node_to_idx = {name: i for i, name in enumerate(group_names)}
    adj = np.zeros((len(group_names), len(group_names)), dtype=np.float32)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for edge in payload.get("edges", []):
            src = edge.get("source")
            dst = edge.get("target")
            weight = float(edge.get("weight", 1.0))
            if src in node_to_idx and dst in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[dst]
                adj[i, j] = max(adj[i, j], weight)
                adj[j, i] = max(adj[j, i], weight)
    else:
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            src = str(row.get("source", ""))
            dst = str(row.get("target", ""))
            weight = float(row.get("weight", 1.0))
            if src in node_to_idx and dst in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[dst]
                adj[i, j] = max(adj[i, j], weight)
                adj[j, i] = max(adj[j, i], weight)
    return normalize_adjacency(torch.from_numpy(adj))


def build_lightweight_kg(
    *,
    binned: pd.DataFrame,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    feature_schema: list[dict] | None,
    split,
    bin_size: str,
    kg_source: str = "lightweight",
    external_kg_path: str | None = None,
    kg_top_k: int = 6,
    kg_min_cooccurrence: int = 2,
    kg_ontology: str = "auto",
) -> KGArtifacts:
    groups = resolve_feature_groups(feat_cols=feat_cols, feature_schema=feature_schema)
    group_names = [group.name for group in groups]
    train_ids = [str(pid) for pid in split.train]
    val_ids = [str(pid) for pid in split.val]

    if kg_source == "external":
        if not external_kg_path:
            raise ValueError("kg_source='external' requires external_kg_path")
        global_adj = _load_external_graph(Path(external_kg_path), group_names)
    else:
        visit_mask = build_group_mask_tensor(
            obs_mask=obs_mask[obs_mask["patient_id"].astype(str).isin(set(train_ids))].copy(),
            groups=groups,
            feat_cols=feat_cols,
        ).reshape(-1, len(groups))
        visit_mask = (visit_mask > 0.0).to(dtype=torch.float32)
        coocc = torch.matmul(visit_mask.transpose(0, 1), visit_mask)
        coocc = torch.where(coocc >= float(kg_min_cooccurrence), coocc, torch.zeros_like(coocc))

        if kg_top_k > 0 and coocc.numel() > 0:
            topk = min(int(kg_top_k), coocc.size(-1))
            vals, idx = torch.topk(coocc, k=topk, dim=-1)
            top_adj = torch.zeros_like(coocc)
            top_adj.scatter_(1, idx, vals)
            coocc = torch.maximum(top_adj, top_adj.transpose(0, 1))

        ontology_adj = torch.zeros_like(coocc)
        buckets = [_ontology_bucket(name, kg_ontology) for name in group_names]
        for i in range(len(buckets)):
            if not buckets[i]:
                continue
            for j in range(i + 1, len(buckets)):
                if buckets[i] == buckets[j]:
                    ontology_adj[i, j] = 1.0
                    ontology_adj[j, i] = 1.0

        global_adj = normalize_adjacency(coocc + ontology_adj)

    def _build_extra(patient_ids: list[str]) -> dict[str, torch.Tensor]:
        subset_binned = binned[binned["patient_id"].astype(str).isin(set(patient_ids))].copy()
        subset_obs = obs_mask[obs_mask["patient_id"].astype(str).isin(set(patient_ids))].copy()
        group_values = build_group_sequence_tensor(
            binned=subset_binned,
            groups=groups,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            reduce="mean",
        )
        max_len = int(group_values.shape[1])
        group_mask = build_group_mask_tensor(
            obs_mask=subset_obs,
            groups=groups,
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        )
        visit_time_map = build_visit_time_map(
            binned=subset_binned,
            patient_ids=set(patient_ids),
            bin_size=bin_size,
        )
        visit_time = build_visit_time_tensor(visit_time_map, patient_ids=patient_ids, max_len=max_len)
        return {
            "group_values": group_values,
            "group_mask": group_mask,
            "visit_time": visit_time,
        }

    return KGArtifacts(
        groups=groups,
        group_names=group_names,
        global_adj=global_adj,
        train_extra=_build_extra(train_ids),
        val_extra=_build_extra(val_ids),
        extra_meta={
            "kg_group_names": group_names,
            "kg_source": kg_source,
        },
    )
