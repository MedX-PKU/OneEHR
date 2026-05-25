"""Deep survival models: DeepSurv and DeepHit.

These models predict time-to-event outcomes with censoring support.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSurv(nn.Module):
    """DeepSurv: Cox proportional hazards deep neural network (Katzman et al., 2018).

    Outputs a single log-risk score per patient. Trained with the Cox
    partial likelihood loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_d, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        # Use last time step of sequence
        if x.ndim == 3:
            idx = (lengths - 1).clamp(min=0).long()
            x = x[torch.arange(x.size(0), device=x.device), idx]
        return self.net(x)


class DeepHit(nn.Module):
    """DeepHit: competing risks survival model (Lee et al., 2018).

    Outputs discrete-time survival probabilities over a fixed set of
    time horizons.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_time_bins: int = 10,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = 1,  # unused, kept for API consistency
    ) -> None:
        super().__init__()
        self.num_time_bins = num_time_bins
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_d, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_d = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_time_bins)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.ndim == 3:
            idx = (lengths - 1).clamp(min=0).long()
            x = x[torch.arange(x.size(0), device=x.device), idx]
        h = self.shared(x)
        # Softmax over time bins gives P(T=t | X)
        return torch.softmax(self.output(h), dim=-1)


# --- Loss Functions ---


class CoxPHLoss(nn.Module):
    """Negative log partial likelihood for Cox PH models."""

    def forward(self, risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        risk_scores : (N,) predicted log-risk
        times : (N,) observed times
        events : (N,) event indicator (1=event, 0=censored)
        """
        # Sort by descending time
        order = torch.argsort(times, descending=True)
        risk = risk_scores[order]
        event = events[order]

        # Breslow approximation of partial likelihood
        log_cumsum = torch.logcumsumexp(risk, dim=0)
        loss = -(risk - log_cumsum) * event
        return loss[event.bool()].mean() if event.sum() > 0 else loss.mean()


class DeepHitLoss(nn.Module):
    """Combined log-likelihood and ranking loss for DeepHit."""

    def __init__(self, alpha: float = 0.5, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(
        self,
        pmf: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pmf : (N, T) predicted probability mass function
        times : (N,) observed times
        events : (N,) event indicator
        time_bins : (T,) bin edges
        """
        # Discretize times to bin indices
        bin_idx = torch.bucketize(times, time_bins) - 1
        bin_idx = bin_idx.clamp(0, pmf.shape[1] - 1)

        # Log-likelihood for events
        eps = 1e-7
        # For events: -log P(T=t)
        # For censored: -log S(t) = -log sum_{k>t} P(T=k)
        nll = torch.zeros(len(times), device=pmf.device)
        for i in range(len(times)):
            if events[i] == 1:
                nll[i] = -torch.log(pmf[i, bin_idx[i]] + eps)
            else:
                surv = pmf[i, bin_idx[i] :].sum()
                nll[i] = -torch.log(surv + eps)

        loss = nll.mean()

        # Ranking loss (concordance-encouraging)
        if self.alpha > 0 and events.sum() > 1:
            cdf = torch.cumsum(pmf, dim=1)
            rank_loss = torch.tensor(0.0, device=pmf.device)
            n_pairs = 0
            event_idx = torch.where(events == 1)[0]
            for i in event_idx:
                later = times > times[i]
                if later.sum() == 0:
                    continue
                diff = cdf[later, bin_idx[i]] - cdf[i, bin_idx[i]]
                rank_loss = rank_loss + torch.exp(diff / self.sigma).mean()
                n_pairs += 1
            if n_pairs > 0:
                loss = loss + self.alpha * rank_loss / n_pairs

        return loss
