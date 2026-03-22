"""PRISM (ATCare) model — missingness-aware, prototype-based EHR model.

All PRISM-specific logic (model classes, data preparation) lives here.
Adapted from prism.py (ATCare).

Model classes:
    PRISMModel      — patient-level prediction (last-visit pooling)
    PRISMTimeModel  — time-level prediction (prefix-rollout)

Data preparation:
    prepare_prism_inputs() — computes dim_list, centers, obs_rates, time_delta
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from oneehr.models.recurrent import last_by_lengths


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────

class AdjacencyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dc_param = nn.Parameter(torch.tensor([0.05]))

    def forward(self, x: torch.Tensor, dc: torch.Tensor) -> torch.Tensor:
        # x, dc: [N, D]
        b, d = x.shape
        x1 = x.unsqueeze(1).expand(-1, b, d)
        x2 = x.unsqueeze(0).expand(b, -1, d)
        dc1 = dc.unsqueeze(1).expand(-1, b, d)
        dc2 = dc.unsqueeze(0).expand(b, -1, d)
        sim = 1 / (
            (1 - self.dc_param) * (x1 - x2) ** 2
            + self.dc_param * torch.exp(1 - dc1) * torch.exp(1 - dc2)
        ).mean(2)
        eye = torch.eye(b, device=x.device)
        return sim * (1 - eye) + eye


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        std = 1.0 / math.sqrt(out_dim)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        y = torch.mm(adj.float(), torch.mm(x.float(), self.weight.float()))
        return y + self.bias.float() if self.bias is not None else y


class SqueezeLayer(nn.Module):
    def __init__(self, dim: int, squeeze_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(F.gelu(self.fc1(x))))


class GroupPatientLearner(nn.Module):
    def __init__(self, num_feat: int, centers: torch.Tensor, hidden_dim: int = 32):
        super().__init__()
        self.num_feat = num_feat
        self.hidden_dim = hidden_dim

        n_clusters, center_dim = centers.shape
        self.register_buffer("_raw_centers", centers.clone())
        self.register_buffer("_proj_centers", torch.zeros(n_clusters, num_feat))
        self._needs_proj = center_dim != num_feat

        if self._needs_proj:
            self.centers_proj = nn.Linear(center_dim, num_feat)
        else:
            self.centers_proj = None
            self._proj_centers.copy_(centers)

        self._proj_done = False

        self.adjacency = AdjacencyLayer()
        self.gcn = GCNLayer(num_feat, hidden_dim)
        self.embed_out = nn.Sequential(nn.Linear(hidden_dim, num_feat), nn.Sigmoid())
        self.center_confidence = nn.Parameter(torch.ones(n_clusters, num_feat))
        self.importance = SqueezeLayer(num_feat, squeeze_dim=16)

    def forward(
        self, x: torch.Tensor, dc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, num_feat, feat_dim], dc: [B, num_feat]
        b = x.shape[0]
        x_avg = x.mean(dim=-1)  # [B, num_feat]

        # Project raw centers once, then use projected version.
        if not self._proj_done and self._needs_proj:
            self._proj_centers = self.centers_proj(self._raw_centers.to(x.device))
            self._proj_done = True
        centers = self._proj_centers.to(x.device)

        x_all = torch.cat([x_avg, centers]).detach()
        dc_all = torch.cat([dc, self.center_confidence.to(dc.device)]).detach()
        adj = self.adjacency(x_all, dc_all)

        group_adj = adj[:b, b:]  # [B, K]
        group_label = group_adj.argmax(dim=1)

        embedding = self.embed_out(self.gcn(x_all, adj))
        group_emb = embedding[b:]  # [K, num_feat]
        group_patient_emb = group_emb[group_label]  # [B, num_feat]

        # Update projected centers (detached) for next forward call.
        self._proj_centers = group_emb.detach()

        group_scores = self.importance(group_emb)[group_label]  # [B, num_feat]
        return group_scores, group_emb, group_patient_emb, group_label


class ConfidenceLearner(nn.Module):
    def __init__(self, num_feat: int):
        super().__init__()
        self.decay_term = nn.Parameter(torch.full((num_feat,), 0.8))
        self.global_missing_param = nn.Parameter(torch.tensor([0.3]))
        self.act = nn.Tanh()
        self.proj = nn.Linear(num_feat, 1)
        self.weight = nn.Linear(num_feat, num_feat)

    def forward(
        self,
        attn: torch.Tensor,   # [B, T, num_feat]
        gfe: torch.Tensor,    # [num_feat]
        td: torch.Tensor,     # [B, T, num_feat]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = attn.shape
        td = td.to(attn.dtype).clamp(max=2.0)
        gfe = gfe.to(attn.dtype).unsqueeze(0).unsqueeze(0).expand(b, t, -1)

        divide = self.decay_term * torch.log(math.e + (1 - attn) * td)
        dc = self.act(attn / divide)
        mask = (td >= 2.0).float()
        dc = dc * (1 - mask) + self.global_missing_param * gfe * mask
        score = torch.sigmoid(self.proj(dc))
        return dc, score


class FeatureAttention(nn.Module):
    def __init__(self, num_feat: int, channel: int, calib: bool = False):
        super().__init__()
        self.query = nn.Linear(channel, channel)
        self.key = nn.Linear(channel, channel)
        self.value = nn.Linear(channel, channel)
        self.calib = calib
        if calib:
            self.confidence_param = nn.Parameter(torch.tensor([0.05]))
            self.confidence_learner = ConfidenceLearner(num_feat)

    def forward(
        self,
        x: torch.Tensor,           # [B, T, num_feat, C]
        gfe: torch.Tensor | None,  # [num_feat]
        td: torch.Tensor | None,   # [B, T, num_feat]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, NF, C = x.size()
        Q = self.query(x[:, -1, :, :])  # [B, NF, C]
        x_flat = x.view(B, -1, C)       # [B, T*NF, C]
        K = self.key(x_flat)
        V = self.value(x_flat)

        attn = torch.softmax(Q.bmm(K.transpose(1, 2)) / (C ** 0.5), dim=2)  # [B, NF, T*NF]
        attn_4d = attn.view(B, NF, T, NF)

        dc = None
        if self.calib and gfe is not None and td is not None:
            dc, _ = self.confidence_learner(attn_4d.mean(dim=1), gfe, td)
            dc_4d = dc.unsqueeze(1).expand(-1, NF, -1, -1)  # [B, NF, T, NF]
            calib_attn = self.confidence_param * dc_4d + (1 - self.confidence_param) * attn_4d
            dc = dc_4d.mean(dim=2)  # [B, NF, NF]
        else:
            calib_attn = attn_4d

        # Reshape back and apply to V
        ca_flat = calib_attn.view(B, NF, T * NF).transpose(1, 2)  # [B, T*NF, NF]
        ca_flat = ca_flat.repeat(1, 1, T)  # [B, T*NF, NF*T]
        out = ca_flat.bmm(V).view(B, T, NF, C)  # [B, T, NF, C]
        return out, dc


class StaticProj(nn.Module):
    """Project all static features into a single feat_dim vector (OneEHR convention)."""

    def __init__(self, static_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(static_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).unsqueeze(1)  # [B, 1, feat_dim]


class MCGRUEncoder(nn.Module):
    """Per-group GRU encoder — each group gets its own GRU."""

    def __init__(self, dim_list: list[int], feat_dim: int = 8):
        super().__init__()
        self.dim_list = dim_list
        self.feat_dim = feat_dim
        self.grus = nn.ModuleList([
            nn.GRU(d, feat_dim, num_layers=1, batch_first=True) for d in dim_list
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        n_feat = len(self.dim_list)
        out = torch.zeros(B, T, n_feat, self.feat_dim, device=x.device, dtype=x.dtype)
        offset = 0
        for i, gru in enumerate(self.grus):
            d = self.dim_list[i]
            out[:, :, i] = gru(x[:, :, offset : offset + d])[0]
            offset += d
        return out


def _reduce_to_groups(
    vals: torch.Tensor, dim_list: list[int],
) -> torch.Tensor:
    """Reduce per-column tensor to per-group by taking first col of each group.

    vals: [*prefix, input_dim]  ->  [*prefix, num_groups]
    """
    prefix = vals.shape[:-1]
    out = torch.zeros(*prefix, len(dim_list), device=vals.device, dtype=vals.dtype)
    offset = 0
    for i, d in enumerate(dim_list):
        out[..., i] = vals[..., offset]
        offset += d
    return out


# ──────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────

class PRISMModel(nn.Module):
    """Patient-level PRISM (ATCare) model.

    Args:
        input_dim: total number of binned feature columns.
        hidden_dim: GRU / projection hidden size.
        feat_dim: per-group GRU output dimension.
        out_dim: output dimension (1 for binary).
        static_dim: dimension of static features (0 = no static branch).
        dim_list: list of ints — size of each feature group for MCGRU.
        centers: [K, input_dim] tensor of prototype cluster centers.
        n_clusters: number of clusters (used only if centers is None).
        calib: whether to use calibrated attention.
        dropout: dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        feat_dim: int = 8,
        out_dim: int = 1,
        static_dim: int = 0,
        dim_list: list[int] | None = None,
        centers: torch.Tensor | None = None,
        n_clusters: int = 10,
        calib: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim

        # Default: one group per column.
        if dim_list is None:
            dim_list = [1] * input_dim
        self.dim_list = dim_list

        num_lab_feat = len(dim_list)
        self.num_lab_feat = num_lab_feat
        has_static = static_dim > 0
        num_feat = num_lab_feat + (1 if has_static else 0)
        self.num_feat = num_feat

        # Encoders
        self.mcgru = MCGRUEncoder(dim_list, feat_dim)
        if has_static:
            self.static_proj = StaticProj(static_dim, feat_dim)
        else:
            self.static_proj = None

        # Default centers (random, will be replaced by prepare_prism_inputs).
        if centers is None:
            centers = torch.randn(n_clusters, input_dim)
        self.group_patient_learner = GroupPatientLearner(num_feat, centers, hidden_dim)
        self.group_embed_param = nn.Parameter(torch.tensor([0.05]))

        # Attention
        self.attention = FeatureAttention(num_feat, feat_dim, calib=calib)

        # Projections
        self.proj1 = nn.Linear(num_feat * feat_dim, num_feat)
        self.proj2 = nn.Linear(num_feat, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Interpretability storage
        self._last_attn = None
        self._last_group_label = None

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        **extra,
    ) -> torch.Tensor:
        obs_rates = extra.get("obs_rates")   # [input_dim] or [num_feat]
        time_delta = extra.get("time_delta")  # [B, T, input_dim] or [B, T, num_feat]

        B, T, D = x.shape

        # Encode lab features through per-group GRU.
        lab = self.mcgru(x)  # [B, T, num_lab_feat, feat_dim]

        # Static branch — project to single feat_dim vector, broadcast across time.
        if self.static_proj is not None and static is not None:
            s = self.static_proj(static)  # [B, 1, feat_dim]
            s = s.expand(-1, T, -1).unsqueeze(2)  # [B, T, 1, feat_dim]
            feat = torch.cat([s, lab], dim=2)  # [B, T, num_feat, feat_dim]
        else:
            feat = lab  # [B, T, num_lab_feat, feat_dim]

        # Reduce obs_rates and time_delta from input_dim to num_feat (group level).
        gfe = None
        td = None
        if obs_rates is not None:
            gfe = obs_rates.to(x.device)
            if gfe.shape[-1] == self.input_dim and self.input_dim != self.num_lab_feat:
                gfe = _reduce_to_groups(gfe, self.dim_list)
            if self.static_proj is not None and gfe.shape[-1] == self.num_lab_feat:
                gfe = torch.cat([torch.ones(1, device=gfe.device), gfe])
        if time_delta is not None:
            td = time_delta.to(x.device)
            if td.shape[-1] == self.input_dim and self.input_dim != self.num_lab_feat:
                td = _reduce_to_groups(td, self.dim_list)
            if self.static_proj is not None and td.shape[-1] == self.num_lab_feat:
                pad = torch.zeros(B, T, 1, device=td.device, dtype=td.dtype)
                td = torch.cat([pad, td], dim=-1)

        # Feature attention (with optional calibration).
        context, dc = self.attention(feat, gfe, td)
        feat = feat + context  # residual

        # Group patient learner — uses last visit.
        last_visit = feat[:, -1, :, :]  # [B, num_feat, feat_dim]
        if dc is not None:
            last_dc = dc[:, -1, :]  # [B, num_feat]
        else:
            last_dc = torch.ones(B, self.num_feat, device=x.device)

        group_scores, group_emb, group_patient_emb, group_label = \
            self.group_patient_learner(last_visit, last_dc)
        self._last_group_label = group_label.detach()

        # Project and mix with group embedding.
        h = feat.flatten(2)  # [B, T, num_feat*feat_dim]
        h = self.proj1(h)    # [B, T, num_feat]
        h = self.group_embed_param * group_patient_emb.unsqueeze(1) + (1 - self.group_embed_param) * h
        h = self.proj2(h)    # [B, T, hidden_dim]
        h = self.dropout(h)

        _, out = self.gru(h)
        out = out.mean(dim=0)  # [B, hidden_dim]
        return self.head(out)  # [B, out_dim]


class PRISMTimeModel(nn.Module):
    """Time-level PRISM: prefix-rollout like ConCareTimeModel."""

    def __init__(self, **kwargs):
        super().__init__()
        out_dim = kwargs.pop("out_dim", 1)
        kwargs["out_dim"] = out_dim
        self.core = PRISMModel(**kwargs)
        # Expose static_dim for has_static_branch()
        self.static_dim = self.core.static_dim

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        **extra,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        outputs = []
        for t in range(T):
            cur_x = x[:, : t + 1, :]
            cur_len = lengths.clamp(max=t + 1).clamp(min=1)
            # Slice time_delta if present.
            cur_extra = {}
            if "obs_rates" in extra:
                cur_extra["obs_rates"] = extra["obs_rates"]
            if "time_delta" in extra:
                cur_extra["time_delta"] = extra["time_delta"][:, : t + 1, :]
            outputs.append(self.core(cur_x, cur_len, static, **cur_extra))
        return torch.stack(outputs, dim=1)


# ──────────────────────────────────────────────────────────────────────
# Data preparation — all PRISM-specific naming stays here
# ──────────────────────────────────────────────────────────────────────

def _cluster(data: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """Simple K-means clustering, returns centers [K, D]."""
    n = data.shape[0]
    k = min(k, n)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=k, replace=False)
    centers = data[idx].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        codes = dists.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = codes == j
            if mask.any():
                new_centers[j] = data[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers


def prepare_prism_inputs(
    binned: pd.DataFrame,
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    feature_schema: list[dict],
    train_pids: set[str],
    *,
    n_clusters: int = 10,
    bin_size: str = "1d",
) -> dict[str, Any]:
    """Compute all PRISM-specific inputs from preprocessing artifacts.

    Returns dict with keys:
        dim_list      — list[int], per-group column counts
        centers       — torch.Tensor [K, input_dim]
        obs_rates     — torch.Tensor [input_dim]  (per-feature training observation rate)
        time_delta    — dict mapping patient_id -> np.ndarray [T, D]
    """
    from oneehr.utils import parse_bin_size

    # 1. dim_list from feature_schema
    dim_list = [entry["dim"] for entry in feature_schema]

    # 2. Observation rates on training split
    mask_cols = [c for c in feat_cols if c in obs_mask.columns]
    train_mask = obs_mask[obs_mask["patient_id"].astype(str).isin(train_pids)]
    obs_rates_np = train_mask[mask_cols].values.mean(axis=0).astype(np.float32)
    obs_rates = torch.from_numpy(obs_rates_np)

    # 3. Time-delta sequences per patient
    # For each patient-visit, compute time since last observation per feature.
    freq = parse_bin_size(bin_size)
    td_unit = pd.Timedelta(freq)

    time_delta_map: dict[str, np.ndarray] = {}
    for pid, grp in obs_mask.groupby("patient_id", sort=False):
        pid_str = str(pid)
        grp = grp.sort_values("bin_time")
        mask_vals = grp[mask_cols].values.astype(np.float32)  # [T, D]
        T, D = mask_vals.shape
        td = np.full((T, D), 2.0, dtype=np.float32)
        last_obs_time = np.full(D, -np.inf)
        times = grp["bin_time"].values

        for t in range(T):
            for d_idx in range(D):
                if mask_vals[t, d_idx] > 0.5:
                    if last_obs_time[d_idx] != -np.inf:
                        gap = (pd.Timestamp(times[t]) - pd.Timestamp(times[int(last_obs_time[d_idx])])) / td_unit
                        td[t, d_idx] = min(float(gap), 2.0)
                    else:
                        td[t, d_idx] = 0.0  # First observation
                    last_obs_time[d_idx] = t
                # else: stays 2.0 (never observed or gap > threshold)
        time_delta_map[pid_str] = td

    # 4. K-means centers from training last-visit feature vectors
    train_binned = binned[binned["patient_id"].astype(str).isin(train_pids)].copy()
    last_visit = (
        train_binned.sort_values(["patient_id", "bin_time"], kind="stable")
        .groupby("patient_id", sort=False)[feat_cols]
        .last()
    )
    data_np = last_visit.values.astype(np.float32)
    centers_np = _cluster(data_np, n_clusters)
    centers = torch.from_numpy(centers_np)

    return {
        "dim_list": dim_list,
        "centers": centers,
        "obs_rates": obs_rates,
        "time_delta_map": time_delta_map,
    }


def build_time_delta_tensor(
    time_delta_map: dict[str, np.ndarray],
    patient_ids: list[str],
    max_len: int,
    input_dim: int,
) -> torch.Tensor:
    """Build padded time-delta tensor [B, T, D] aligned with patient sequences."""
    B = len(patient_ids)
    td = torch.full((B, max_len, input_dim), 2.0)
    for i, pid in enumerate(patient_ids):
        arr = time_delta_map.get(str(pid))
        if arr is not None:
            t = min(arr.shape[0], max_len)
            td[i, :t, :] = torch.from_numpy(arr[:t])
    return td


def _build_time_delta_map(
    obs_mask: pd.DataFrame,
    feat_cols: list[str],
    *,
    bin_size: str = "1d",
    patient_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    from oneehr.utils import parse_bin_size

    if patient_ids is not None:
        obs_mask = obs_mask[obs_mask["patient_id"].astype(str).isin(patient_ids)].copy()

    mask_cols = [c for c in feat_cols if c in obs_mask.columns]
    freq = parse_bin_size(bin_size)
    td_unit = pd.Timedelta(freq)

    time_delta_map: dict[str, np.ndarray] = {}
    for pid, grp in obs_mask.groupby("patient_id", sort=False):
        pid_str = str(pid)
        grp = grp.sort_values("bin_time")
        mask_vals = grp[mask_cols].values.astype(np.float32)
        T, D = mask_vals.shape
        td = np.full((T, D), 2.0, dtype=np.float32)
        last_obs_time = np.full(D, -np.inf)
        times = grp["bin_time"].values

        for t in range(T):
            for d_idx in range(D):
                if mask_vals[t, d_idx] > 0.5:
                    if last_obs_time[d_idx] != -np.inf:
                        gap = (pd.Timestamp(times[t]) - pd.Timestamp(times[int(last_obs_time[d_idx])])) / td_unit
                        td[t, d_idx] = min(float(gap), 2.0)
                    else:
                        td[t, d_idx] = 0.0
                    last_obs_time[d_idx] = t
        time_delta_map[pid_str] = td

    return time_delta_map


def prepare_prism_training_artifacts(
    *,
    model_cfg,
    binned: pd.DataFrame,
    feat_cols: list[str],
    split,
    run_dir,
    preprocess_cfg,
) -> dict[str, object]:
    from oneehr.data.sequence import build_patient_sequences

    feature_schema_path = run_dir / "preprocess" / "feature_schema.json"
    obs_mask_path = run_dir / "preprocess" / "obs_mask.parquet"
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    obs_mask_df = pd.read_parquet(obs_mask_path)

    train_pids = set(str(p) for p in split.train)
    val_pids = set(str(p) for p in split.val)
    prism_data = prepare_prism_inputs(
        binned,
        obs_mask_df,
        feat_cols,
        feature_schema,
        train_pids,
        n_clusters=int(model_cfg.params.get("n_clusters", 10)),
        bin_size=preprocess_cfg.bin_size,
    )

    def _build_extra(patient_ids: set[str]) -> dict[str, torch.Tensor]:
        subset = binned[binned["patient_id"].astype(str).isin(patient_ids)].copy()
        seq_pids, _, lens = build_patient_sequences(subset, feat_cols)
        max_len = int(lens.max()) if len(lens) else 0
        return {
            "obs_rates": prism_data["obs_rates"],
            "time_delta": build_time_delta_tensor(
                prism_data["time_delta_map"],
                list(seq_pids),
                max_len,
                len(feat_cols),
            ),
        }

    return {
        "model_cfg": type(model_cfg)(
            name=model_cfg.name,
            params={**model_cfg.params, "dim_list": prism_data["dim_list"], "centers": prism_data["centers"]},
        ),
        "train_extra": _build_extra(train_pids),
        "val_extra": _build_extra(val_pids),
        "extra_meta": {"obs_rates": prism_data["obs_rates"].tolist()},
    }


def prepare_prism_inference_extra(
    *,
    meta: dict,
    run_dir,
    feat_cols: list[str],
    patient_ids: list[str],
    max_len: int,
) -> dict[str, torch.Tensor]:
    obs_rates_list = meta.get("extra", {}).get("obs_rates")
    if obs_rates_list is None:
        return {}

    from oneehr.artifacts.manifest import read_manifest

    manifest = read_manifest(run_dir)
    bin_size = manifest.get("config", {}).get("preprocess", {}).get("bin_size", "1d")
    obs_mask_path = run_dir / "preprocess" / "obs_mask.parquet"
    obs_mask_df = pd.read_parquet(obs_mask_path)
    td_map = _build_time_delta_map(
        obs_mask_df,
        feat_cols,
        bin_size=bin_size,
        patient_ids=set(str(pid) for pid in patient_ids),
    )

    return {
        "obs_rates": torch.tensor(obs_rates_list, dtype=torch.float32),
        "time_delta": build_time_delta_tensor(td_map, patient_ids, max_len, len(feat_cols)),
    }
