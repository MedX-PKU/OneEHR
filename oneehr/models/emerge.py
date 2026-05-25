"""EMERGE multimodal EHR/text fusion models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch import nn

from oneehr.config.schema import ModelConfig
from oneehr.models.recurrent import last_by_lengths


class EhrEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        ehr_net: str = "gru",
        *,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.25,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.ehr_net = ehr_net
        if ehr_net == "gru":
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif ehr_net == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif ehr_net == "rnn":
            self.encoder = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif ehr_net == "transformer":
            self.proj = nn.Linear(input_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ffn_multiplier,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        elif ehr_net == "modern_transformer":
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.positional_encoding = SinusoidalPositionEncoding(hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ffn_multiplier,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unsupported EMERGE EHR encoder: {ehr_net}")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.size(1))

        if self.ehr_net == "transformer":
            encoded = self.encoder(self.proj(x), src_key_padding_mask=_padding_mask(lengths, x.size(1)))
            return last_by_lengths(encoded, lengths)

        if self.ehr_net == "modern_transformer":
            projected = self.proj(x)
            cls = self.cls_token.expand(projected.size(0), -1, -1)
            tokens = self.positional_encoding(torch.cat([cls, projected], dim=1))
            padding = _padding_mask(lengths, x.size(1))
            cls_padding = torch.zeros((x.size(0), 1), dtype=torch.bool, device=x.device)
            encoded = self.encoder(tokens, src_key_padding_mask=torch.cat([cls_padding, padding], dim=1))
            return self.norm(encoded[:, 0, :])

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / self.hidden_dim)
        )
        pe = torch.zeros(seq_len, self.hidden_dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return x + pe.unsqueeze(0)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        self.text_to_ehr = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ehr_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ehr = nn.LayerNorm(hidden_dim)
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, ehr: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        ehr_seq = ehr.unsqueeze(1)
        text_seq = text.unsqueeze(1)
        z_ehr, _ = self.text_to_ehr(query=text_seq, key=ehr_seq, value=ehr_seq)
        z_text, _ = self.ehr_to_text(query=ehr_seq, key=text_seq, value=text_seq)
        z_ehr = self.norm_ehr(z_ehr + text_seq).squeeze(1)
        z_text = self.norm_text(z_text + ehr_seq).squeeze(1)
        return self.ffn(torch.cat([z_ehr, z_text], dim=-1))


class MAGGate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, 1)
        self.adjust = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Parameter(torch.ones(()))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, base: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.gate(torch.cat([base, aux], dim=-1)))
        adjust = self.adjust(weight * aux)
        scale = torch.norm(base, dim=-1, keepdim=True) / torch.clamp(
            torch.norm(adjust, dim=-1, keepdim=True), min=1e-6
        )
        alpha = torch.clamp(scale * self.beta, max=1.0)
        return self.dropout(self.norm(base + alpha * adjust))


class TensorFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear((hidden_dim + 1) * (hidden_dim + 1), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch_size = first.size(0)
        ones = torch.ones((batch_size, 1), device=first.device, dtype=first.dtype)
        first = torch.cat([first, ones], dim=-1)
        second = torch.cat([second, ones], dim=-1)
        outer = torch.bmm(first.unsqueeze(2), second.unsqueeze(1))
        return self.net(outer.flatten(1))


class Concatenation(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([first, second], dim=-1))


class TokenTransformerFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.25,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_multiplier,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pool = nn.Linear(hidden_dim, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        ehr: torch.Tensor,
        note: torch.Tensor,
        summary: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack([ehr, note, summary], dim=1) + self.modality_embedding
        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        scores = self.pool(encoded).squeeze(-1).masked_fill(~token_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        residual = (tokens * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.norm(self.out(pooled) + residual)


class EMERGE(nn.Module):
    def __init__(
        self,
        input_ehr_dim: int,
        input_note_dim: int = 768,
        input_summary_dim: int = 768,
        hidden_dim: int = 128,
        ehr_net: str = "gru",
        text_fusion: str = "concat",
        modality_fusion: str = "ours",
        use_modality: str = "ehr_note_summary",
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_multiplier: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.text_fusion = text_fusion
        self.modality_fusion = modality_fusion
        self.use_modality = use_modality

        self.ehr_encoder = EhrEncoder(
            input_ehr_dim,
            hidden_dim,
            ehr_net=ehr_net,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
        )
        self.note_proj = nn.Linear(input_note_dim, hidden_dim)
        self.summary_proj = nn.Linear(input_summary_dim, hidden_dim)
        self.text_concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.text_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.text_alpha = nn.Parameter(torch.tensor(0.5))
        self.text_mag = MAGGate(hidden_dim, dropout=dropout)

        if modality_fusion == "ours":
            self.fusion = CrossAttentionFusion(hidden_dim, num_heads=num_heads, dropout=dropout)
        elif modality_fusion == "token_transformer":
            self.fusion = TokenTransformerFusion(
                hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                ffn_multiplier=ffn_multiplier,
            )
        elif modality_fusion == "mag":
            self.fusion = MAGGate(hidden_dim, dropout=dropout)
        elif modality_fusion == "concat":
            self.fusion = Concatenation(hidden_dim)
        elif modality_fusion == "tf":
            self.fusion = TensorFusion(hidden_dim)
        else:
            raise ValueError(f"Unsupported EMERGE modality fusion: {modality_fusion}")

    def forward(
        self,
        x_ehr: torch.Tensor,
        lengths: torch.Tensor | None = None,
        note_embedding: torch.Tensor | None = None,
        summary_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ehr_embedding = self.ehr_encoder(x_ehr, lengths)
        if self.use_modality == "ehr_only":
            return ehr_embedding

        note = self._project_note(note_embedding) if self._uses_note else None
        summary = self._project_summary(summary_embedding) if self._uses_summary else None

        if self.use_modality == "note_only":
            return _require_projected(note, "note_embedding")
        if self.use_modality == "summary_only":
            return _require_projected(summary, "summary_embedding")

        text_embedding = self._text_embedding(note, summary)
        if self.use_modality == "note_summary":
            return text_embedding

        if self.modality_fusion == "token_transformer":
            token_mask = self._modality_mask(x_ehr.size(0), x_ehr.device)
            zero = torch.zeros_like(ehr_embedding)
            return self.fusion(
                ehr_embedding,
                note if note is not None else zero,
                summary if summary is not None else zero,
                token_mask,
            )
        return self.fusion(ehr_embedding, text_embedding)

    @property
    def _uses_note(self) -> bool:
        return self.use_modality in {"ehr_note_summary", "ehr_note", "note_only", "note_summary"}

    @property
    def _uses_summary(self) -> bool:
        return self.use_modality in {"ehr_note_summary", "ehr_summary", "summary_only", "note_summary"}

    def _project_note(self, note_embedding: torch.Tensor | None) -> torch.Tensor:
        if note_embedding is None:
            raise ValueError("EMERGE requires `note_embedding` in forward() for the selected modality")
        return self.note_proj(note_embedding)

    def _project_summary(self, summary_embedding: torch.Tensor | None) -> torch.Tensor:
        if summary_embedding is None:
            raise ValueError("EMERGE requires `summary_embedding` in forward() for the selected modality")
        return self.summary_proj(summary_embedding)

    def _text_embedding(self, note: torch.Tensor | None, summary: torch.Tensor | None) -> torch.Tensor:
        if self.use_modality == "ehr_note":
            return _require_projected(note, "note_embedding")
        if self.use_modality == "ehr_summary":
            return _require_projected(summary, "summary_embedding")

        note = _require_projected(note, "note_embedding")
        summary = _require_projected(summary, "summary_embedding")
        if self.text_fusion == "note_only":
            return note
        if self.text_fusion == "summary_only":
            return summary
        if self.text_fusion == "add":
            return note + summary
        if self.text_fusion == "concat":
            return self.text_concat_proj(torch.cat([note, summary], dim=-1))
        if self.text_fusion == "gated":
            gate = torch.sigmoid(self.text_gate(torch.cat([note, summary], dim=-1)))
            return gate * note + (1.0 - gate) * summary
        if self.text_fusion == "adaptive":
            alpha = torch.clamp(self.text_alpha, 0.0, 1.0)
            return alpha * note + (1.0 - alpha) * summary
        if self.text_fusion == "mag":
            return self.text_mag(note, summary)
        raise ValueError(f"Unsupported EMERGE text fusion: {self.text_fusion}")

    def _modality_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        active = {
            "ehr_note_summary": (True, True, True),
            "ehr_note": (True, True, False),
            "ehr_summary": (True, False, True),
        }.get(self.use_modality, (True, True, True))
        return torch.tensor(active, device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1)


class EMERGEModel(nn.Module):
    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 128,
        out_dim: int = 1,
        *,
        input_ehr_dim: int | None = None,
        input_note_dim: int = 768,
        input_summary_dim: int = 768,
        ehr_net: str = "gru",
        text_fusion: str = "concat",
        modality_fusion: str = "ours",
        use_modality: str = "ehr_note_summary",
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_multiplier: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        resolved_input_dim = _resolve_input_dim(input_dim, input_ehr_dim)
        self.backbone = EMERGE(
            input_ehr_dim=resolved_input_dim,
            input_note_dim=input_note_dim,
            input_summary_dim=input_summary_dim,
            hidden_dim=hidden_dim,
            ehr_net=ehr_net,
            text_fusion=text_fusion,
            modality_fusion=modality_fusion,
            use_modality=use_modality,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        note_embedding: torch.Tensor | None = None,
        summary_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del static
        embedding = self.backbone(
            x,
            lengths,
            note_embedding=note_embedding,
            summary_embedding=summary_embedding,
        )
        return self.head(embedding)


class EMERGETimeModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.core = EMERGEModel(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        note_embedding: torch.Tensor | None = None,
        summary_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for t in range(x.size(1)):
            cur_lengths = lengths.clamp(max=t + 1).clamp(min=1)
            outputs.append(
                self.core(
                    x[:, : t + 1, :],
                    cur_lengths,
                    static,
                    note_embedding=note_embedding,
                    summary_embedding=summary_embedding,
                )
            )
        return torch.stack(outputs, dim=1)


class EMERGEPredictor(EMERGEModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("out_dim", 2)
        if "input_ehr_dim" in kwargs and "input_dim" not in kwargs:
            kwargs["input_dim"] = kwargs.pop("input_ehr_dim")
        super().__init__(**kwargs)

    def forward(
        self,
        x_ehr: torch.Tensor,
        x_note_embedding: torch.Tensor,
        x_summary_embedding: torch.Tensor,
    ) -> torch.Tensor:
        lengths = torch.full((x_ehr.size(0),), x_ehr.size(1), dtype=torch.long, device=x_ehr.device)
        return super().forward(
            x_ehr,
            lengths,
            note_embedding=x_note_embedding,
            summary_embedding=x_summary_embedding,
        )


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


def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


def _resolve_input_dim(input_dim: int | None, input_ehr_dim: int | None) -> int:
    if input_dim is None and input_ehr_dim is None:
        raise ValueError("EMERGE requires `input_dim`")
    if input_dim is not None and input_ehr_dim is not None and int(input_dim) != int(input_ehr_dim):
        raise ValueError("EMERGE received conflicting `input_dim` and `input_ehr_dim` values")
    return int(input_dim if input_dim is not None else input_ehr_dim)


def _require_projected(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"EMERGE requires `{name}` in forward() for the selected modality")
    return value
