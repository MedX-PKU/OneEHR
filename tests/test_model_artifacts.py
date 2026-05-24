"""Tests for model-side artifacts and auxiliary tensor alignment."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch


def test_align_extra_to_patient_ids_reorders_batched_tensors():
    from oneehr.training.trainer import _align_extra_to_patient_ids

    extra = {
        "_patient_ids": ["p3", "p1", "p2"],
        "group_values": torch.tensor([[[3.0]], [[1.0]], [[2.0]]]),
        "global_vector": torch.tensor([9.0]),
    }

    aligned = _align_extra_to_patient_ids(extra, ["p1", "p2"])
    assert aligned["_patient_ids"] == ["p1", "p2"]
    assert aligned["group_values"].squeeze().tolist() == [1.0, 2.0]
    assert aligned["global_vector"].tolist() == [9.0]


def test_external_kg_matches_medical_code_aliases(tmp_path):
    from oneehr.models.kg import build_lightweight_kg

    binned = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "bin_time": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "DX_ICD9_25000": [1.0, 0.0],
            "DX_ICD10_E119": [0.0, 1.0],
        }
    )
    obs_mask = binned.copy()
    obs_mask[["DX_ICD9_25000", "DX_ICD10_E119"]] = 1.0
    kg_path = tmp_path / "kg.csv"
    pd.DataFrame(
        {
            "source": ["ICD9:25000", "ICD10:E119", "missing"],
            "target": ["ICD10:E119", "ICD9:25000", "ICD9:25000"],
            "weight": [1.0, 1.0, 1.0],
        }
    ).to_csv(kg_path, index=False)

    kg = build_lightweight_kg(
        binned=binned,
        obs_mask=obs_mask,
        feat_cols=["DX_ICD9_25000", "DX_ICD10_E119"],
        feature_schema=None,
        split=SimpleNamespace(train=["p2", "p1"], val=["p1"]),
        bin_size="1d",
        kg_source="external",
        external_kg_path=str(kg_path),
    )

    coverage = kg.extra_meta["kg_coverage"]
    assert coverage["input_edge_count"] == 3
    assert coverage["matched_edge_count"] == 2
    assert coverage["ignored_edge_count"] == 1
    assert kg.global_adj.shape == (2, 2)
    assert kg.train_extra["_patient_ids"] == ["p2", "p1"]
