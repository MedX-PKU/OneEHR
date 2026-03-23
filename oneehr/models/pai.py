"""PAI on top of a GRU backbone.

PAI (Learnable Prompt as Pseudo-Imputation) is implemented here as a GRU model
that replaces truly-missing feature positions with a learned prompt vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths


def _sorted_feature_frame(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    return (
        df[["patient_id", "bin_time", *feat_cols]]
        .copy()
        .sort_values(
            ["patient_id", "bin_time"],
            kind="stable",
        )
    )


def build_missing_mask_tensor(
    *,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: list[str] | None = None,
    max_len: int | None = None,
) -> torch.Tensor:
    from oneehr.data.sequence import build_patient_sequences

    obs = _sorted_feature_frame(obs_mask, feat_cols)
    seq_pids, seqs, _ = build_patient_sequences(obs, feat_cols)
    seq_map = {str(pid): 1.0 - seq.astype(np.float32, copy=False) for pid, seq in zip(seq_pids, seqs, strict=True)}

    if patient_ids is None:
        patient_ids = list(seq_pids)
    patient_ids = [str(pid) for pid in patient_ids]

    if max_len is None:
        max_len = max((seq_map.get(pid, np.zeros((0, len(feat_cols)), dtype=np.float32)).shape[0] for pid in patient_ids), default=0)

    out = np.zeros((len(patient_ids), max_len, len(feat_cols)), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        seq = seq_map.get(pid)
        if seq is None:
            continue
        steps = min(seq.shape[0], max_len)
        out[i, :steps, :] = seq[:steps]
    return torch.from_numpy(out)


def _compute_prompt_init_values(
    *,
    binned: pd.DataFrame,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: set[str],
    prompt_init: str,
) -> torch.Tensor:
    prompt_init = str(prompt_init).lower()
    if prompt_init == "zero":
        return torch.zeros(len(feat_cols), dtype=torch.float32)
    if prompt_init == "random":
        return torch.empty(len(feat_cols), dtype=torch.float32).uniform_(-0.05, 0.05)
    if prompt_init != "median":
        raise ValueError(f"Unsupported PAI prompt_init={prompt_init!r}")

    feat = _sorted_feature_frame(
        binned[binned["patient_id"].astype(str).isin(patient_ids)],
        feat_cols,
    )
    obs = _sorted_feature_frame(
        obs_mask[obs_mask["patient_id"].astype(str).isin(patient_ids)],
        feat_cols,
    )
    merged = feat.merge(
        obs,
        on=["patient_id", "bin_time"],
        how="inner",
        suffixes=("", "__obs"),
    )
    medians = []
    for col in feat_cols:
        obs_col = f"{col}__obs"
        valid = merged[obs_col] > 0.5
        if valid.any():
            medians.append(float(merged.loc[valid, col].median()))
        else:
            medians.append(0.0)
    return torch.tensor(medians, dtype=torch.float32)


def prepare_pai_training_artifacts(
    *,
    model_cfg,
    binned: pd.DataFrame,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    split,
) -> dict[str, object]:
    params = dict(model_cfg.params)
    prompt_init = str(params.get("prompt_init", "median"))
    train_pids = set(str(pid) for pid in split.train)
    val_pids = set(str(pid) for pid in split.val)

    if "prompt_init_values" not in params:
        params["prompt_init_values"] = _compute_prompt_init_values(
            binned=binned,
            obs_mask=obs_mask,
            feat_cols=feat_cols,
            patient_ids=train_pids,
            prompt_init=prompt_init,
        )

    train_obs = obs_mask[obs_mask["patient_id"].astype(str).isin(train_pids)].copy()
    val_obs = obs_mask[obs_mask["patient_id"].astype(str).isin(val_pids)].copy()

    return {
        "model_cfg": type(model_cfg)(name=model_cfg.name, params=params),
        "train_extra": {"missing_mask": build_missing_mask_tensor(obs_mask=train_obs, feat_cols=feat_cols)},
        "val_extra": {"missing_mask": build_missing_mask_tensor(obs_mask=val_obs, feat_cols=feat_cols)},
        "extra_meta": None,
    }


def build_pai_inference_extra(
    *,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    patient_ids: list[str],
    max_len: int,
) -> dict[str, torch.Tensor]:
    return {
        "missing_mask": build_missing_mask_tensor(
            obs_mask=obs_mask[obs_mask["patient_id"].astype(str).isin(set(str(pid) for pid in patient_ids))].copy(),
            feat_cols=feat_cols,
            patient_ids=patient_ids,
            max_len=max_len,
        ),
    }


class PAIEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        prompt_init_values: torch.Tensor | None = None,
    ):
        super().__init__()
        init_values = torch.zeros(input_dim, dtype=torch.float32)
        if prompt_init_values is not None:
            init_values = prompt_init_values.detach().to(dtype=torch.float32).reshape(input_dim)
        self.prompt = nn.Parameter(init_values.clone())
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def apply_prompt(self, x: torch.Tensor, missing_mask: torch.Tensor | None) -> torch.Tensor:
        if missing_mask is None:
            raise ValueError("PAI requires `missing_mask` in forward()")
        if missing_mask.shape != x.shape:
            raise ValueError(f"PAI missing_mask shape mismatch: expected {tuple(x.shape)}, got {tuple(missing_mask.shape)}")
        mask = missing_mask.to(device=x.device, dtype=x.dtype)
        prompt = self.prompt.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        return x * (1.0 - mask) + prompt * mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, missing_mask: torch.Tensor | None) -> torch.Tensor:
        x_prompted = self.apply_prompt(x, missing_mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_prompted,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out


class PAIModel(nn.Module):
    """Patient-level PAI with a GRU backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        prompt_init_values: torch.Tensor | None = None,
    ):
        super().__init__()
        self.encoder = PAIEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            prompt_init_values=prompt_init_values,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encoder(x, lengths, missing_mask)
        last = last_by_lengths(h, lengths)
        return self.head(last)


class PAITimeModel(nn.Module):
    """Time-level PAI via prefix rollout."""

    def __init__(self, **kwargs):
        super().__init__()
        self.core = PAIModel(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        missing_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if missing_mask is None:
            raise ValueError("PAI requires `missing_mask` in forward()")
        outputs = []
        for t in range(x.size(1)):
            cur_x = x[:, : t + 1, :]
            cur_lengths = lengths.clamp(max=t + 1).clamp(min=1)
            cur_missing = missing_mask[:, : t + 1, :]
            outputs.append(self.core(cur_x, cur_lengths, static, missing_mask=cur_missing))
        return torch.stack(outputs, dim=1)
