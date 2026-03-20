"""DrAgent model for patient-level and time-level prediction.

Paper: Junyi Gao et al. Dr. Agent: Clinical predictive model via mimicked
second opinions. JAMIA.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oneehr.models.recurrent import last_by_lengths


class AgentLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        static_dim: int = 0,
        n_actions: int = 10,
        n_units: int = 64,
        dropout: float = 0.5,
        lamda: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim
        self.n_actions = n_actions
        self.lamda = lamda
        self.dropout_rate = dropout

        # Agent networks
        self.agent1_fc1 = nn.Linear(hidden_dim + static_dim, n_units)
        self.agent2_fc1 = nn.Linear(input_dim + static_dim, n_units)
        self.agent1_fc2 = nn.Linear(n_units, n_actions)
        self.agent2_fc2 = nn.Linear(n_units, n_actions)

        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        for name, p in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(p)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if static_dim > 0:
            self.init_h = nn.Linear(static_dim, hidden_dim)
            self.fusion = nn.Linear(hidden_dim + static_dim, hidden_dim)

    def _choose_action(self, obs: torch.Tensor, fc1: nn.Linear, fc2: nn.Linear) -> torch.Tensor:
        h = torch.tanh(fc1(obs.detach()))
        logits = fc2(h)
        if self.training:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1)
        return logits.argmax(dim=-1, keepdim=True)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        device = x.device

        if static is not None and self.static_dim > 0:
            cur_h = self.init_h(static)
        else:
            cur_h = torch.zeros(B, self.hidden_dim, device=device)

        observed_h = torch.zeros(self.n_actions, B, self.hidden_dim, device=device)
        h_list = []

        for t in range(T):
            cur_input = x[:, t, :]

            if t == 0:
                action_h = cur_h
            else:
                observed_h = torch.cat([observed_h[1:], cur_h.unsqueeze(0)], 0)
                obs_mean = observed_h.mean(dim=0)

                obs1 = torch.cat([obs_mean, static], dim=1) if static is not None and self.static_dim > 0 else obs_mean
                obs2 = torch.cat([cur_input, static], dim=1) if static is not None and self.static_dim > 0 else cur_input

                # Pad obs1 to match expected input dim if no static
                idx1 = self._choose_action(obs1, self.agent1_fc1, self.agent1_fc2).long()
                idx2 = self._choose_action(obs2, self.agent2_fc1, self.agent2_fc2).long()

                batch_idx = torch.arange(B, device=device).unsqueeze(-1)
                h1 = observed_h[idx1, batch_idx].squeeze(1)
                h2 = observed_h[idx2, batch_idx].squeeze(1)
                action_h = (h1 + h2) / 2

            weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h
            cur_h = self.rnn(cur_input, weighted_h)
            h_list.append(cur_h)

        h = torch.stack(h_list, dim=1)  # (B, T, H)

        if static is not None and self.static_dim > 0:
            s_expand = static.unsqueeze(1).expand(-1, T, -1)
            h = self.fusion(torch.cat([h, s_expand], dim=-1))

        return h  # (B, T, H)


class DrAgentModel(nn.Module):
    """Patient-level DrAgent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        static_dim: int = 0,
        n_actions: int = 10,
        n_units: int = 64,
        dropout: float = 0.5,
        lamda: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.layer = AgentLayer(input_dim, hidden_dim, static_dim, n_actions, n_units, dropout, lamda)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.layer(x, lengths, static)
        last = last_by_lengths(h, lengths)
        return self.head(self.dropout(last))


class DrAgentTimeModel(nn.Module):
    """Time-level DrAgent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        static_dim: int = 0,
        n_actions: int = 10,
        n_units: int = 64,
        dropout: float = 0.5,
        lamda: float = 0.5,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.layer = AgentLayer(input_dim, hidden_dim, static_dim, n_actions, n_units, dropout, lamda)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.layer(x, lengths, static)
        return self.head(self.dropout(h))
