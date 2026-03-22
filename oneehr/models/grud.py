"""GRU-D models for missing-aware EHR sequence modeling."""

from __future__ import annotations

import torch
from torch import nn

from oneehr.models.recurrent import last_by_lengths


class GRUDBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        feature_means: torch.Tensor | None = None,
    ):
        super().__init__()
        init_means = torch.zeros(input_dim, dtype=torch.float32)
        if feature_means is not None:
            init_means = feature_means.detach().to(dtype=torch.float32).reshape(input_dim)
        self.register_buffer("feature_means", init_means)
        self.x_decay = nn.Linear(input_dim, input_dim)
        self.h_decay = nn.Linear(input_dim, hidden_dim)
        self.cell = nn.GRUCell(input_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        *,
        missing_mask: torch.Tensor | None,
        time_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        if missing_mask is None or time_delta is None:
            raise ValueError("GRU-D requires missing_mask and time_delta in forward()")
        if missing_mask.shape != x.shape or time_delta.shape != x.shape:
            raise ValueError(
                "GRU-D expects missing_mask and time_delta to match x shape "
                f"{tuple(x.shape)}, got {tuple(missing_mask.shape)} and {tuple(time_delta.shape)}"
            )

        batch_size, steps, input_dim = x.shape
        observed = 1.0 - missing_mask.to(dtype=x.dtype, device=x.device)
        delta = time_delta.to(dtype=x.dtype, device=x.device)
        means = self.feature_means.to(dtype=x.dtype, device=x.device).view(1, -1)

        h = x.new_zeros((batch_size, self.h_decay.out_features))
        x_last = means.expand(batch_size, -1).clone()
        outputs = []

        for t in range(steps):
            valid = (t < lengths).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            m_t = observed[:, t, :]
            d_t = delta[:, t, :]
            x_t = x[:, t, :]

            gamma_x = torch.exp(-torch.relu(self.x_decay(d_t)))
            gamma_h = torch.exp(-torch.relu(self.h_decay(d_t)))

            x_last_candidate = m_t * x_t + (1.0 - m_t) * x_last
            x_hat = m_t * x_t + (1.0 - m_t) * (gamma_x * x_last + (1.0 - gamma_x) * means)
            h_tilde = gamma_h * h
            h_candidate = self.cell(torch.cat([x_hat, m_t], dim=-1), h_tilde)
            h_candidate = self.dropout(h_candidate)

            h = valid * h_candidate + (1.0 - valid) * h
            x_last = valid * x_last_candidate + (1.0 - valid) * x_last
            outputs.append(h * valid)

        return torch.stack(outputs, dim=1)


class GRUDModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.0,
        feature_means: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = GRUDBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            feature_means=feature_means,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        missing_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        visit_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.backbone(
            x,
            lengths,
            missing_mask=missing_mask,
            time_delta=time_delta,
        )
        return self.head(last_by_lengths(h, lengths))


class GRUDTimeModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        out_dim = int(kwargs.get("out_dim", 1))
        self.backbone = GRUDBackbone(
            input_dim=int(kwargs["input_dim"]),
            hidden_dim=hidden_dim,
            dropout=float(kwargs.get("dropout", 0.0)),
            feature_means=kwargs.get("feature_means"),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
        *,
        missing_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
        visit_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(
            self.backbone(
                x,
                lengths,
                missing_mask=missing_mask,
                time_delta=time_delta,
            )
        )
