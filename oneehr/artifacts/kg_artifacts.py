"""Lightweight knowledge-graph helpers for graph-enhanced EHR models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oneehr.artifacts.model_policy import (
    DEFAULT_KG_MIN_COOCCURRENCE,
    DEFAULT_KG_ONTOLOGY,
    DEFAULT_KG_SOURCE,
    DEFAULT_KG_TOP_K,
    resolve_kg_preset,
)
from oneehr.artifacts.tensor_adapters import (
    FeatureGroup,
    build_group_mask_tensor,
    build_group_sequence_tensor,
    build_visit_time_map,
    build_visit_time_tensor,
    resolve_feature_groups,
)
from oneehr.medcode.features import feature_code_aliases, ontology_bucket
from oneehr.models.layers.graph import normalize_adjacency


@dataclass(frozen=True)
class KGArtifacts:
    groups: list[FeatureGroup]
    group_names: list[str]
    global_adj: torch.Tensor
    train_extra: dict[str, object]
    val_extra: dict[str, object]
    extra_meta: dict[str, object]


@dataclass(frozen=True)
class KGCoverage:
    """Coverage metadata for a graph projected onto model feature groups."""

    node_count: int
    input_edge_count: int
    matched_edge_count: int
    ignored_edge_count: int
    nonzero_edge_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "node_count": self.node_count,
            "input_edge_count": self.input_edge_count,
            "matched_edge_count": self.matched_edge_count,
            "ignored_edge_count": self.ignored_edge_count,
            "nonzero_edge_count": self.nonzero_edge_count,
        }


@dataclass(frozen=True)
class KGGraph:
    """Feature-group graph consumed by OneEHR KG baselines."""

    group_names: list[str]
    adjacency: torch.Tensor
    coverage: KGCoverage


def _group_alias_index(group_names: list[str]) -> dict[str, int]:
    alias_to_idx: dict[str, int] = {}
    for idx, name in enumerate(group_names):
        for alias in feature_code_aliases(name):
            alias_to_idx.setdefault(alias, idx)
            alias_to_idx.setdefault(alias.upper(), idx)
    return alias_to_idx


def _read_external_edges(path: Path) -> list[tuple[str, str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"External KG not found: {path}")

    if path.suffix.lower() == ".json":
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("edges", [])
        edges = []
        for edge in rows:
            src = edge.get("source", edge.get("head"))
            dst = edge.get("target", edge.get("tail"))
            if src is None or dst is None:
                continue
            edges.append((str(src), str(dst), float(edge.get("weight", 1.0))))
        return edges

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    if {"source", "target"}.issubset(frame.columns):
        src_col, dst_col = "source", "target"
    elif {"head", "tail"}.issubset(frame.columns):
        src_col, dst_col = "head", "tail"
    else:
        src_col, dst_col = frame.columns[:2]

    weight_col = "weight" if "weight" in frame.columns else None
    return [
        (
            str(row[src_col]),
            str(row[dst_col]),
            float(row[weight_col]) if weight_col is not None and not pd.isna(row[weight_col]) else 1.0,
        )
        for _, row in frame.iterrows()
    ]


def _load_external_graph(path: Path, group_names: list[str]) -> KGGraph:
    node_to_idx = _group_alias_index(group_names)
    adj = np.zeros((len(group_names), len(group_names)), dtype=np.float32)
    edges = _read_external_edges(path)
    matched = 0
    for src, dst, weight in edges:
        src_idx = node_to_idx.get(src, node_to_idx.get(src.upper()))
        dst_idx = node_to_idx.get(dst, node_to_idx.get(dst.upper()))
        if src_idx is None or dst_idx is None:
            continue
        matched += 1
        adj[src_idx, dst_idx] = max(adj[src_idx, dst_idx], weight)
        adj[dst_idx, src_idx] = max(adj[dst_idx, src_idx], weight)

    coverage = KGCoverage(
        node_count=len(group_names),
        input_edge_count=len(edges),
        matched_edge_count=matched,
        ignored_edge_count=len(edges) - matched,
        nonzero_edge_count=int((adj > 0).sum()),
    )
    return KGGraph(group_names=group_names, adjacency=normalize_adjacency(torch.from_numpy(adj)), coverage=coverage)


def _build_lightweight_graph(
    *,
    visit_mask: torch.Tensor,
    group_names: list[str],
    kg_top_k: int,
    kg_min_cooccurrence: int,
    kg_ontology: str,
) -> KGGraph:
    coocc = torch.matmul(visit_mask.transpose(0, 1), visit_mask)
    coocc = torch.where(coocc >= float(kg_min_cooccurrence), coocc, torch.zeros_like(coocc))

    if kg_top_k > 0 and coocc.numel() > 0:
        topk = min(int(kg_top_k), coocc.size(-1))
        vals, idx = torch.topk(coocc, k=topk, dim=-1)
        top_adj = torch.zeros_like(coocc)
        top_adj.scatter_(1, idx, vals)
        coocc = torch.maximum(top_adj, top_adj.transpose(0, 1))

    ontology_adj = torch.zeros_like(coocc)
    buckets = [ontology_bucket(name, kg_ontology) for name in group_names]
    for i in range(len(buckets)):
        if not buckets[i]:
            continue
        for j in range(i + 1, len(buckets)):
            if buckets[i] == buckets[j]:
                ontology_adj[i, j] = 1.0
                ontology_adj[j, i] = 1.0

    adj = coocc + ontology_adj
    coverage = KGCoverage(
        node_count=len(group_names),
        input_edge_count=int((visit_mask.sum(dim=1) > 1).sum().item()),
        matched_edge_count=int((coocc > 0).sum().item()),
        ignored_edge_count=0,
        nonzero_edge_count=int((adj > 0).sum().item()),
    )
    return KGGraph(group_names=group_names, adjacency=normalize_adjacency(adj), coverage=coverage)


def build_lightweight_kg(
    *,
    binned: pd.DataFrame,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    feature_schema: list[dict] | None,
    split,
    bin_size: str,
    kg_source: str = DEFAULT_KG_SOURCE,
    external_kg_path: str | None = None,
    kg_top_k: int = DEFAULT_KG_TOP_K,
    kg_min_cooccurrence: int = DEFAULT_KG_MIN_COOCCURRENCE,
    kg_ontology: str = DEFAULT_KG_ONTOLOGY,
) -> KGArtifacts:
    groups = resolve_feature_groups(feat_cols=feat_cols, feature_schema=feature_schema)
    group_names = [group.name for group in groups]
    train_ids = [str(pid) for pid in split.train]
    val_ids = [str(pid) for pid in split.val]
    if kg_source not in {"lightweight", "external"}:
        raise ValueError(f"Unsupported kg_source={kg_source!r}. Expected 'lightweight' or 'external'.")

    if kg_source == "external":
        if not external_kg_path:
            raise ValueError("kg_source='external' requires external_kg_path")
        graph = _load_external_graph(Path(external_kg_path), group_names)
    else:
        visit_mask = build_group_mask_tensor(
            obs_mask=obs_mask[obs_mask["patient_id"].astype(str).isin(set(train_ids))].copy(),
            groups=groups,
            feat_cols=feat_cols,
        ).reshape(-1, len(groups))
        visit_mask = (visit_mask > 0.0).to(dtype=torch.float32)
        graph = _build_lightweight_graph(
            visit_mask=visit_mask,
            group_names=group_names,
            kg_top_k=kg_top_k,
            kg_min_cooccurrence=kg_min_cooccurrence,
            kg_ontology=kg_ontology,
        )

    def _build_extra(patient_ids: list[str]) -> dict[str, object]:
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
            "_patient_ids": list(patient_ids),
            "group_values": group_values,
            "group_mask": group_mask,
            "visit_time": visit_time,
        }

    kg_config = {
        "kg_source": kg_source,
        "kg_top_k": int(kg_top_k),
        "kg_min_cooccurrence": int(kg_min_cooccurrence),
        "kg_ontology": str(kg_ontology),
    }
    if external_kg_path:
        kg_config["external_kg_path"] = str(external_kg_path)

    return KGArtifacts(
        groups=groups,
        group_names=group_names,
        global_adj=graph.adjacency,
        train_extra=_build_extra(train_ids),
        val_extra=_build_extra(val_ids),
        extra_meta={
            "kg_group_names": group_names,
            "kg_source": kg_source,
            "kg_preset": resolve_kg_preset(
                kg_source=kg_source,
                kg_top_k=kg_top_k,
                kg_min_cooccurrence=kg_min_cooccurrence,
                kg_ontology=kg_ontology,
            ),
            "kg_config": kg_config,
            "kg_coverage": graph.coverage.as_dict(),
        },
    )
