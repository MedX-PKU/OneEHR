"""Model-side artifact and external-asset policy.

This module is intentionally small and dependency-free so model defaults,
runtime preparation, docs, and tests can refer to one shared contract.
"""

from __future__ import annotations

from dataclasses import dataclass

KG_MODEL_NAMES: frozenset[str] = frozenset({"graphcare", "kerprint", "protoehr"})

DEFAULT_KG_SOURCE = "lightweight"
DEFAULT_KG_PRESET = "lightweight_auto"
DEFAULT_KG_TOP_K = 6
DEFAULT_KG_MIN_COOCCURRENCE = 2
DEFAULT_KG_ONTOLOGY = "auto"
DEFAULT_LIGHTWEIGHT_KG_PARAMS: dict[str, object] = {
    "kg_source": DEFAULT_KG_SOURCE,
    "kg_top_k": DEFAULT_KG_TOP_K,
    "kg_min_cooccurrence": DEFAULT_KG_MIN_COOCCURRENCE,
    "kg_ontology": DEFAULT_KG_ONTOLOGY,
}


@dataclass(frozen=True)
class ModelArtifactPolicy:
    """Reproducibility contract for model-specific assets and derived tensors."""

    model_name: str
    requires_external_pretraining: bool = False
    required_external_assets: tuple[str, ...] = ()
    optional_external_assets: tuple[str, ...] = ()
    required_preprocess_artifacts: tuple[str, ...] = ()
    derived_train_artifacts: tuple[str, ...] = ()
    default_kg_preset: str | None = None
    note: str = "Trains from scratch using the run config and training split."

    @property
    def has_reproducibility_metadata(self) -> bool:
        return bool(self.requires_external_pretraining or self.required_external_assets or self.optional_external_assets or self.required_preprocess_artifacts or self.derived_train_artifacts or self.default_kg_preset)

    def as_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "requires_external_pretraining": self.requires_external_pretraining,
            "required_external_assets": list(self.required_external_assets),
            "optional_external_assets": list(self.optional_external_assets),
            "required_preprocess_artifacts": list(self.required_preprocess_artifacts),
            "derived_train_artifacts": list(self.derived_train_artifacts),
            "default_kg_preset": self.default_kg_preset,
            "note": self.note,
        }


def resolve_kg_preset(
    *,
    kg_source: str,
    kg_top_k: int,
    kg_min_cooccurrence: int,
    kg_ontology: str,
) -> str:
    """Return a stable preset name for KG provenance metadata."""

    if kg_source == "external":
        return "external"
    if kg_source == DEFAULT_KG_SOURCE and int(kg_top_k) == DEFAULT_KG_TOP_K and int(kg_min_cooccurrence) == DEFAULT_KG_MIN_COOCCURRENCE and str(kg_ontology) == DEFAULT_KG_ONTOLOGY:
        return DEFAULT_KG_PRESET
    return "lightweight_custom"


def model_artifact_policy(model_name: str) -> ModelArtifactPolicy:
    """Return the reproducibility policy for a model name."""

    name = str(model_name).lower()
    return _MODEL_ARTIFACT_POLICIES.get(name, ModelArtifactPolicy(model_name=name))


def checkpoint_artifact_meta(model_name: str, extra_meta: dict[str, object] | None = None) -> dict[str, object] | None:
    """Attach model artifact policy to checkpoint metadata when relevant."""

    meta = dict(extra_meta or {})
    policy = model_artifact_policy(model_name)
    if policy.has_reproducibility_metadata:
        meta["artifact_policy"] = policy.as_dict()
    return meta or None


_TIME_EXTRA_ARTIFACTS = ("missing_mask", "time_delta", "visit_time")

_MODEL_ARTIFACT_POLICIES: dict[str, ModelArtifactPolicy] = {
    "grud": ModelArtifactPolicy(
        model_name="grud",
        required_preprocess_artifacts=("preprocess/obs_mask.parquet",),
        derived_train_artifacts=("feature_means", *_TIME_EXTRA_ARTIFACTS),
        note="GRU-D trains from scratch; feature means, missing masks, and time deltas are derived from the training split.",
    ),
    "pai": ModelArtifactPolicy(
        model_name="pai",
        required_preprocess_artifacts=("preprocess/obs_mask.parquet",),
        derived_train_artifacts=("prompt_init_values", "missing_mask"),
        note="PAI trains from scratch; prompt initialization values are derived from the training split unless supplied explicitly.",
    ),
    "emerge": ModelArtifactPolicy(
        model_name="emerge",
        optional_external_assets=("note_text_path", "summary_text_path", "note_embedding_path", "summary_embedding_path"),
        required_preprocess_artifacts=("preprocess/binned.parquet",),
        derived_train_artifacts=("note_embedding", "summary_embedding", "preprocess/emerge_text_embeddings.pt"),
        note="EMERGE trains from scratch; note and summary embeddings are supplied as explicit artifacts or deterministically derived from run data through the shared text artifact interface.",
    ),
    "prism": ModelArtifactPolicy(
        model_name="prism",
        required_preprocess_artifacts=("preprocess/feature_schema.json", "preprocess/obs_mask.parquet"),
        derived_train_artifacts=("dim_list", "centers", "obs_rates", "time_delta"),
        note="PRISM trains from scratch; cluster centers and observation-rate statistics are deterministic run artifacts.",
    ),
    "safari": ModelArtifactPolicy(
        model_name="safari",
        required_preprocess_artifacts=("preprocess/feature_schema.json",),
        derived_train_artifacts=("dim_list",),
        note="SAFARI trains from scratch; feature-group dimensions are resolved from preprocessing artifacts.",
    ),
    "lsan": ModelArtifactPolicy(
        model_name="lsan",
        required_preprocess_artifacts=("preprocess/feature_schema.json",),
        derived_train_artifacts=("group_indices", "group_names"),
        note="LSAN trains from scratch; feature groups are resolved from preprocessing artifacts.",
    ),
    "hitanet": ModelArtifactPolicy(
        model_name="hitanet",
        required_preprocess_artifacts=("preprocess/feature_schema.json", "preprocess/obs_mask.parquet"),
        derived_train_artifacts=("group_indices", "group_names", *_TIME_EXTRA_ARTIFACTS),
        note="HiTANet trains from scratch; feature groups and temporal masks are derived from preprocessing artifacts.",
    ),
    "mtand": ModelArtifactPolicy(
        model_name="mtand",
        required_preprocess_artifacts=("preprocess/obs_mask.parquet",),
        derived_train_artifacts=_TIME_EXTRA_ARTIFACTS,
        note="mTAND trains from scratch; relative-time tensors are derived from the run artifacts.",
    ),
    "raindrop": ModelArtifactPolicy(
        model_name="raindrop",
        required_preprocess_artifacts=("preprocess/obs_mask.parquet",),
        derived_train_artifacts=_TIME_EXTRA_ARTIFACTS,
        note="Raindrop trains from scratch; temporal masks and visit times are derived from the run artifacts.",
    ),
    "contiformer": ModelArtifactPolicy(
        model_name="contiformer",
        required_preprocess_artifacts=("preprocess/obs_mask.parquet",),
        derived_train_artifacts=_TIME_EXTRA_ARTIFACTS,
        note="ContiFormer trains from scratch; time-delta tensors are derived from the run artifacts.",
    ),
    "teco": ModelArtifactPolicy(
        model_name="teco",
        required_preprocess_artifacts=("preprocess/obs_mask.parquet",),
        derived_train_artifacts=_TIME_EXTRA_ARTIFACTS,
        note="TECO trains from scratch; temporal tensors are derived from the run artifacts.",
    ),
    **{
        name: ModelArtifactPolicy(
            model_name=name,
            optional_external_assets=("external_kg_path when kg_source='external'",),
            required_preprocess_artifacts=("preprocess/feature_schema.json", "preprocess/obs_mask.parquet"),
            derived_train_artifacts=("group_indices", "group_names", "global_adj", "group_values", "group_mask", "visit_time"),
            default_kg_preset=DEFAULT_KG_PRESET,
            note=f"{name} trains from scratch; the default KG is built from train-split co-occurrence and medcode ontology hints.",
        )
        for name in KG_MODEL_NAMES
    },
}
