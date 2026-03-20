"""StageNet model for patient-level and time-level prediction.

Paper: Junyi Gao et al. Stagenet: Stage-aware neural networks for health risk
prediction. WWW 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oneehr.models.recurrent import last_by_lengths


class StageNetLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 128,
        levels: int = 3,
        conv_size: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = chunk_size * levels
        self.chunk_size = chunk_size
        self.levels = levels
        self.conv_size = conv_size
        self.dropout_rate = dropout

        self.kernel = nn.Linear(input_dim + 1, self.hidden_dim * 4 + levels * 2)
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(self.hidden_dim + 1, self.hidden_dim * 4 + levels * 2)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(self.hidden_dim, self.hidden_dim // 6)
        self.nn_rescale = nn.Linear(self.hidden_dim // 6, self.hidden_dim)
        self.nn_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, conv_size, 1)

        self.dropconnect = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropconnect_r = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.nn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.nn_dropres = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _cumax(x: torch.Tensor, mode: str = "l2r") -> torch.Tensor:
        if mode == "l2r":
            return torch.cumsum(torch.softmax(x, dim=-1), dim=-1)
        x = torch.flip(x, [-1])
        return torch.flip(torch.cumsum(torch.softmax(x, dim=-1), dim=-1), [-1])

    def _step(self, inp, c_last, h_last, interval):
        device = inp.device
        interval = interval.unsqueeze(-1)
        x1 = self.dropconnect(self.kernel(torch.cat([inp, interval], dim=-1)))
        x2 = self.dropconnect_r(self.recurrent_kernel(torch.cat([h_last, interval], dim=-1)))
        x_out = x1 + x2

        f_master = self._cumax(x_out[:, :self.levels], "l2r").unsqueeze(2)
        i_master = self._cumax(x_out[:, self.levels:self.levels * 2], "r2l").unsqueeze(2)

        gates = x_out[:, self.levels * 2:].reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = torch.sigmoid(gates[:, :self.levels])
        i_gate = torch.sigmoid(gates[:, self.levels:self.levels * 2])
        o_gate = torch.sigmoid(gates[:, self.levels * 2:self.levels * 3])
        c_in = torch.tanh(gates[:, self.levels * 3:])

        c_last_r = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master * i_master
        c_out = overlap * (f_gate * c_last_r + i_gate * c_in) + \
                (f_master - overlap) * c_last_r + (i_master - overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)

        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master[..., 0], i_master[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        device = x.device
        time_intervals = torch.ones(B, T, device=device)

        c_out = torch.zeros(B, self.hidden_dim, device=device)
        h_out = torch.zeros(B, self.hidden_dim, device=device)

        tmp_h = torch.zeros(self.conv_size, B, self.hidden_dim, device=device)
        tmp_dis = torch.zeros(self.conv_size, B, device=device)

        h_list = []
        origin_h_list = []

        for t in range(T):
            out, c_out, h_out = self._step(x[:, t, :], c_out, h_out, time_intervals[:, t])
            cur_dist = 1 - out[:, self.hidden_dim:self.hidden_dim + self.levels].mean(-1)
            origin_h_list.append(out[:, :self.hidden_dim])

            tmp_h = torch.cat([tmp_h[1:], out[:, :self.hidden_dim].unsqueeze(0)], 0)
            tmp_dis = torch.cat([tmp_dis[1:], cur_dist.unsqueeze(0)], 0)

            local_dis = torch.softmax(torch.cumsum(tmp_dis.permute(1, 0), dim=1), dim=1)
            local_h = tmp_h.permute(1, 2, 0) * local_dis.unsqueeze(1)

            local_theme = torch.sigmoid(self.nn_rescale(torch.relu(
                self.nn_scale(local_h.mean(dim=-1))
            )))
            local_h = self.nn_conv(local_h).squeeze(-1)
            h_list.append(local_theme * local_h)

        origin_h = torch.stack(origin_h_list, dim=1)
        rnn_out = torch.stack(h_list, dim=1)
        rnn_out = rnn_out + self.nn_dropres(origin_h)
        rnn_out = self.nn_dropout(rnn_out)
        return rnn_out  # (B, T, hidden_dim)


class StageNetModel(nn.Module):
    """Patient-level StageNet."""

    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 128,
        levels: int = 3,
        conv_size: int = 10,
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dim = chunk_size * levels
        self.layer = StageNetLayer(input_dim, chunk_size, levels, conv_size, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.layer(x, lengths)
        last = last_by_lengths(out, lengths)
        return self.head(last)


class StageNetTimeModel(nn.Module):
    """Time-level StageNet."""

    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 128,
        levels: int = 3,
        conv_size: int = 10,
        out_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dim = chunk_size * levels
        self.layer = StageNetLayer(input_dim, chunk_size, levels, conv_size, dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.layer(x, lengths)
        return self.head(out)
