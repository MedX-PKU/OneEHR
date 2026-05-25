"""Tests for model-side artifacts and auxiliary tensor alignment."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
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


def test_lightweight_kg_records_recommended_default_preset():
    from oneehr.models.kg import build_lightweight_kg

    binned = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "bin_time": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "DX_ICD9_25000": [1.0, 1.0],
            "DX_ICD9_4019": [1.0, 1.0],
        }
    )
    obs_mask = binned.copy()
    obs_mask[["DX_ICD9_25000", "DX_ICD9_4019"]] = 1.0

    kg = build_lightweight_kg(
        binned=binned,
        obs_mask=obs_mask,
        feat_cols=["DX_ICD9_25000", "DX_ICD9_4019"],
        feature_schema=None,
        split=SimpleNamespace(train=["p1", "p2"], val=["p2"]),
        bin_size="1d",
    )

    assert kg.extra_meta["kg_preset"] == "lightweight_auto"
    assert kg.extra_meta["kg_config"] == {
        "kg_source": "lightweight",
        "kg_top_k": 6,
        "kg_min_cooccurrence": 2,
        "kg_ontology": "auto",
    }


def test_emerge_text_artifacts_are_split_aligned(tmp_path):
    from oneehr.config.schema import ModelConfig
    from oneehr.models.emerge import build_emerge_inference_extra, prepare_emerge_text_artifacts

    run_dir = tmp_path / "run"
    binned = pd.DataFrame(
        {
            "patient_id": ["p2", "p1", "p1", "p3"],
            "bin_time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-01"]),
            "num__heart_rate": [80.0, 90.0, 95.0, 0.0],
            "num__creatinine": [1.1, 0.0, 1.3, 0.0],
        }
    )

    spec = prepare_emerge_text_artifacts(
        model_cfg=ModelConfig(name="emerge", params={"text_embedding_dim": 8}),
        binned=binned,
        feat_cols=["num__heart_rate", "num__creatinine"],
        split=SimpleNamespace(train=["p1", "p2"], val=["p3"], test=["p2"]),
        run_dir=run_dir,
    )

    assert spec["model_cfg"].params["input_note_dim"] == 8
    assert spec["model_cfg"].params["input_summary_dim"] == 8
    assert spec["train_extra"]["_patient_ids"] == ["p1", "p2"]
    assert spec["train_extra"]["note_embedding"].shape == (2, 8)
    assert (run_dir / "preprocess" / "emerge_text_embeddings.pt").exists()

    infer = build_emerge_inference_extra(
        meta={"extra": spec["extra_meta"]},
        run_dir=run_dir,
        patient_ids=["p2", "p3"],
    )
    assert infer["_patient_ids"] == ["p2", "p3"]
    assert infer["note_embedding"].shape == (2, 8)


def test_emerge_text_artifacts_accept_precomputed_embeddings(tmp_path):
    from oneehr.config.schema import ModelConfig
    from oneehr.models.emerge import prepare_emerge_text_artifacts

    note_path = tmp_path / "note.csv"
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "e0": [1.0, 2.0],
            "e1": [3.0, 4.0],
        }
    ).to_csv(note_path, index=False)
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "embedding": [json.dumps([0.1, 0.2, 0.3]), json.dumps([0.4, 0.5, 0.6])],
        }
    ).to_csv(summary_path, index=False)
    binned = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "bin_time": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "num__x": [0.0, 1.0],
        }
    )

    spec = prepare_emerge_text_artifacts(
        model_cfg=ModelConfig(
            name="emerge",
            params={
                "note_embedding_path": str(note_path),
                "summary_embedding_path": str(summary_path),
            },
        ),
        binned=binned,
        feat_cols=["num__x"],
        split=SimpleNamespace(train=["p2"], val=["p1"], test=[]),
        run_dir=tmp_path / "run",
    )

    assert spec["model_cfg"].params["input_note_dim"] == 2
    assert spec["model_cfg"].params["input_summary_dim"] == 3
    np.testing.assert_allclose(spec["train_extra"]["note_embedding"].numpy(), [[2.0, 4.0]])
    np.testing.assert_allclose(spec["val_extra"]["summary_embedding"].numpy(), [[0.1, 0.2, 0.3]])


def test_model_artifact_policy_exposes_no_external_pretraining():
    from oneehr.models.artifact_policy import checkpoint_artifact_meta, model_artifact_policy, resolve_kg_preset

    graphcare = model_artifact_policy("graphcare")
    assert graphcare.requires_external_pretraining is False
    assert graphcare.required_external_assets == ()
    assert graphcare.optional_external_assets == ("external_kg_path when kg_source='external'",)
    assert graphcare.default_kg_preset == "lightweight_auto"

    grud_meta = checkpoint_artifact_meta("grud")
    assert grud_meta is not None
    assert grud_meta["artifact_policy"]["requires_external_pretraining"] is False
    assert "feature_means" in grud_meta["artifact_policy"]["derived_train_artifacts"]

    assert (
        resolve_kg_preset(
            kg_source="lightweight",
            kg_top_k=4,
            kg_min_cooccurrence=2,
            kg_ontology="auto",
        )
        == "lightweight_custom"
    )
