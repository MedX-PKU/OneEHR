"""Text artifact helpers for multimodal EHR baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from oneehr.config.schema import ModelConfig


@dataclass
class TfidfSvdTextEncoder:
    embedding_dim: int = 768
    max_features: int = 30000
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=int(self.max_features),
            ngram_range=self.ngram_range,
            lowercase=True,
            strip_accents="unicode",
            min_df=1,
        )
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> TfidfSvdTextEncoder:
        clean = _ensure_nonempty_texts(texts)
        matrix = self.vectorizer.fit_transform(clean)
        max_components = min(int(self.embedding_dim), max(1, matrix.shape[0] - 1), max(1, matrix.shape[1] - 1))
        if max_components >= 2:
            self.svd = TruncatedSVD(n_components=max_components, random_state=42)
            self.svd.fit(matrix)
        else:
            self.svd = None
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        clean = _ensure_nonempty_texts(texts)
        matrix = self.vectorizer.transform(clean)
        if self.svd is not None:
            emb = self.svd.transform(matrix).astype(np.float32)
        else:
            emb = matrix.toarray().astype(np.float32)
        emb = normalize(emb, norm="l2", axis=1, copy=False)
        return _pad_or_trim(emb, int(self.embedding_dim))

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


def prepare_emerge_text_artifacts(
    *,
    model_cfg: ModelConfig,
    binned: pd.DataFrame,
    feat_cols: list[str],
    split,
    run_dir: Path,
) -> dict[str, object]:
    """Prepare patient-level note and summary embeddings for EMERGE."""

    params = dict(model_cfg.params)
    if str(params.get("use_modality", "ehr_note_summary")) == "ehr_only":
        return {"model_cfg": model_cfg, "train_extra": None, "val_extra": None, "extra_meta": None}

    patient_ids = _all_patient_ids(binned, split)
    train_ids = [str(pid) for pid in split.train]
    val_ids = [str(pid) for pid in split.val]
    note_dim = int(params.get("input_note_dim", params.get("text_embedding_dim", 768)))
    summary_dim = int(params.get("input_summary_dim", params.get("text_embedding_dim", note_dim)))

    note_embeddings, note_meta = _resolve_embedding_matrix(
        patient_ids=patient_ids,
        binned=binned,
        feat_cols=feat_cols,
        params=params,
        kind="note",
        embedding_dim=note_dim,
        train_ids=train_ids,
    )
    summary_embeddings, summary_meta = _resolve_embedding_matrix(
        patient_ids=patient_ids,
        binned=binned,
        feat_cols=feat_cols,
        params=params,
        kind="summary",
        embedding_dim=summary_dim,
        train_ids=train_ids,
    )

    artifact_rel = Path("preprocess") / "emerge_text_embeddings.pt"
    artifact_path = run_dir / artifact_rel
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "patient_ids": patient_ids,
            "note_embedding": torch.from_numpy(note_embeddings.astype(np.float32, copy=False)),
            "summary_embedding": torch.from_numpy(summary_embeddings.astype(np.float32, copy=False)),
            "config": {
                "note_dim": int(note_embeddings.shape[1]),
                "summary_dim": int(summary_embeddings.shape[1]),
                "note": note_meta,
                "summary": summary_meta,
            },
        },
        artifact_path,
    )

    updated_params = {
        **params,
        "input_note_dim": int(note_embeddings.shape[1]),
        "input_summary_dim": int(summary_embeddings.shape[1]),
    }
    return {
        "model_cfg": type(model_cfg)(name=model_cfg.name, params=updated_params),
        "train_extra": _slice_text_extra(patient_ids, note_embeddings, summary_embeddings, train_ids),
        "val_extra": _slice_text_extra(patient_ids, note_embeddings, summary_embeddings, val_ids),
        "extra_meta": {
            "emerge_text_artifact": artifact_rel.as_posix(),
            "emerge_text_config": {
                "note_dim": int(note_embeddings.shape[1]),
                "summary_dim": int(summary_embeddings.shape[1]),
                "note": note_meta,
                "summary": summary_meta,
            },
        },
    }


def build_emerge_inference_extra(
    *,
    meta: dict[str, Any],
    run_dir: Path,
    patient_ids: list[str],
) -> dict[str, object]:
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    artifact_rel = extra.get("emerge_text_artifact")
    if not artifact_rel:
        return {}
    artifact = torch.load(run_dir / str(artifact_rel), map_location="cpu", weights_only=False)
    source_ids = [str(pid) for pid in artifact["patient_ids"]]
    note_embedding = _as_numpy(artifact["note_embedding"])
    summary_embedding = _as_numpy(artifact["summary_embedding"])
    return _slice_text_extra(source_ids, note_embedding, summary_embedding, [str(pid) for pid in patient_ids])


def _resolve_embedding_matrix(
    *,
    patient_ids: list[str],
    binned: pd.DataFrame,
    feat_cols: list[str],
    params: dict[str, object],
    kind: str,
    embedding_dim: int,
    train_ids: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    embedding_path = params.get(f"{kind}_embedding_path")
    if embedding_path:
        matrix = _load_embedding_matrix(Path(str(embedding_path)), patient_ids, kind=kind)
        return matrix, {"source": "precomputed", "path": str(embedding_path)}

    texts = _resolve_patient_texts(
        patient_ids=patient_ids,
        binned=binned,
        feat_cols=feat_cols,
        path=params.get(f"{kind}_text_path"),
        text_col=str(params.get(f"{kind}_text_col", "text")),
        kind=kind,
    )
    backend = str(params.get("text_embedding_backend", "tfidf_svd"))
    if backend != "tfidf_svd":
        raise ValueError(f"Unsupported EMERGE text_embedding_backend={backend!r}")

    train_texts = [texts.get(pid, "") for pid in train_ids]
    all_texts = [texts.get(pid, "") for pid in patient_ids]
    encoder = TfidfSvdTextEncoder(
        embedding_dim=int(embedding_dim),
        max_features=int(params.get("text_max_features", 30000)),
    ).fit(train_texts)
    matrix = encoder.transform(all_texts)
    return matrix, {
        "source": "text",
        "backend": backend,
        "text_path": str(params.get(f"{kind}_text_path", "")),
        "embedding_dim": int(embedding_dim),
    }


def _resolve_patient_texts(
    *,
    patient_ids: list[str],
    binned: pd.DataFrame,
    feat_cols: list[str],
    path: object | None,
    text_col: str,
    kind: str,
) -> dict[str, str]:
    if path:
        return _read_patient_texts(Path(str(path)), text_col=text_col)
    if kind == "summary":
        return _build_ehr_summary_texts(binned=binned, feat_cols=feat_cols, patient_ids=patient_ids)
    return _build_ehr_note_texts(binned=binned, feat_cols=feat_cols, patient_ids=patient_ids)


def _read_patient_texts(path: Path, *, text_col: str) -> dict[str, str]:
    frame = _read_table(path)
    if "patient_id" not in frame.columns:
        raise ValueError(f"{path} must contain a patient_id column")
    if text_col not in frame.columns:
        raise ValueError(f"{path} must contain text column {text_col!r}")
    frame = frame[["patient_id", text_col]].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame.drop_duplicates("patient_id", keep="last").set_index("patient_id")[text_col].fillna("").astype(str).to_dict()


def _load_embedding_matrix(path: Path, patient_ids: list[str], *, kind: str) -> np.ndarray:
    if path.suffix.lower() == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source_ids = [str(pid) for pid in payload["patient_ids"]]
        key = f"{kind}_embedding" if f"{kind}_embedding" in payload else "embedding"
        matrix = _as_numpy(payload[key])
        return _align_matrix(source_ids, matrix, patient_ids)

    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        source_ids = [str(pid) for pid in payload["patient_ids"]]
        key = f"{kind}_embedding" if f"{kind}_embedding" in payload else "embedding"
        return _align_matrix(source_ids, payload[key].astype(np.float32), patient_ids)

    frame = _read_table(path)
    if "patient_id" not in frame.columns:
        raise ValueError(f"{path} must contain a patient_id column")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if "embedding" in frame.columns:
        values = np.vstack([_parse_embedding_cell(value) for value in frame["embedding"]]).astype(np.float32)
        source_ids = frame["patient_id"].tolist()
        return _align_matrix(source_ids, values, patient_ids)

    value_cols = [col for col in frame.columns if col != "patient_id"]
    if not value_cols:
        raise ValueError(f"{path} must contain embedding columns")
    values = frame[value_cols].to_numpy(dtype=np.float32)
    return _align_matrix(frame["patient_id"].tolist(), values, patient_ids)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Text artifact not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported text artifact format: {path}")


def _build_ehr_note_texts(
    *,
    binned: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: list[str],
) -> dict[str, str]:
    out = {pid: "" for pid in patient_ids}
    if binned.empty:
        return out
    frame = binned[["patient_id", "bin_time", *feat_cols]].copy().sort_values(["patient_id", "bin_time"], kind="stable")
    for pid, group in frame.groupby("patient_id", sort=False):
        pid = str(pid)
        last = group.iloc[-1]
        lines = [f"Patient {pid}.", f"Number of time points: {len(group)}."]
        if "bin_time" in group.columns:
            lines.append(f"Time range: {group['bin_time'].iloc[0]} to {group['bin_time'].iloc[-1]}.")
        latest = []
        for col in feat_cols:
            val = last[col]
            if pd.notna(val) and float(val) != 0.0:
                latest.append(f"{_clean_feature_name(col)}={val}")
        if latest:
            lines.append("Latest nonzero features: " + "; ".join(latest[:64]) + ".")
        out[pid] = " ".join(lines)
    return out


def _build_ehr_summary_texts(
    *,
    binned: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: list[str],
) -> dict[str, str]:
    out = {pid: "" for pid in patient_ids}
    if binned.empty:
        return out
    frame = binned[["patient_id", "bin_time", *feat_cols]].copy().sort_values(["patient_id", "bin_time"], kind="stable")
    for pid, group in frame.groupby("patient_id", sort=False):
        pid = str(pid)
        nonzero_counts = (group[feat_cols].fillna(0.0).to_numpy(dtype=np.float32) != 0.0).sum(axis=0)
        ranked = sorted(
            ((feat_cols[i], int(count)) for i, count in enumerate(nonzero_counts) if count > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        pieces = [f"Patient {pid} has {len(group)} recorded time points."]
        if ranked:
            pieces.append("Most recurrent observed signals: " + ", ".join(f"{_clean_feature_name(name)} ({count})" for name, count in ranked[:24]) + ".")
        else:
            pieces.append("No nonzero dynamic feature signal was recorded after preprocessing.")
        out[pid] = " ".join(pieces)
    return out


def _slice_text_extra(
    source_ids: list[str],
    note_embedding: np.ndarray,
    summary_embedding: np.ndarray,
    patient_ids: list[str],
) -> dict[str, object]:
    index = _index_for_patient_ids(source_ids, patient_ids)
    return {
        "_patient_ids": list(patient_ids),
        "note_embedding": torch.from_numpy(note_embedding[index].astype(np.float32, copy=False)),
        "summary_embedding": torch.from_numpy(summary_embedding[index].astype(np.float32, copy=False)),
    }


def _index_for_patient_ids(source_ids: list[str], patient_ids: list[str]) -> np.ndarray:
    source_pos = {str(pid): idx for idx, pid in enumerate(source_ids)}
    missing = [str(pid) for pid in patient_ids if str(pid) not in source_pos]
    if missing:
        raise ValueError(f"Text embeddings missing patient IDs: {missing[:5]}")
    return np.asarray([source_pos[str(pid)] for pid in patient_ids], dtype=np.int64)


def _align_matrix(source_ids: list[str], matrix: np.ndarray, patient_ids: list[str]) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional")
    index = _index_for_patient_ids([str(pid) for pid in source_ids], [str(pid) for pid in patient_ids])
    return matrix[index].astype(np.float32, copy=False)


def _all_patient_ids(binned: pd.DataFrame, split) -> list[str]:
    ordered = [str(pid) for pid in binned["patient_id"].astype(str).drop_duplicates().tolist()] if "patient_id" in binned.columns else []
    for part in (getattr(split, "train", []), getattr(split, "val", []), getattr(split, "test", [])):
        for pid in part:
            pid = str(pid)
            if pid not in ordered:
                ordered.append(pid)
    return ordered


def _parse_embedding_cell(value: object) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return np.asarray(json.loads(text), dtype=np.float32)
        return np.asarray([float(part) for part in text.replace(";", ",").split(",") if part.strip()], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _ensure_nonempty_texts(texts: list[str]) -> list[str]:
    clean = [" ".join(str(text).split()) for text in texts]
    return [text if text else "none" for text in clean]


def _pad_or_trim(matrix: np.ndarray, dim: int) -> np.ndarray:
    if matrix.shape[1] == dim:
        return matrix.astype(np.float32)
    if matrix.shape[1] > dim:
        return matrix[:, :dim].astype(np.float32)
    pad_width = dim - matrix.shape[1]
    return np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)


def _clean_feature_name(name: str) -> str:
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("cat__"):
        return name[5:]
    return name
