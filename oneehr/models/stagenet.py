from __future__ import annotations

import torch
from torch import nn


from oneehr.models.utils import last_by_lengths


class StageNetLayer(nn.Module):
    """StageNet core layer."""

    def __init__(
        self,
        input_dim: int,
        *,
        chunk_size: int = 128,
        conv_size: int = 10,
        levels: int = 3,
        dropconnect: float = 0.3,
        dropout: float = 0.3,
        dropres: float = 0.3,
    ):
        super().__init__()

        self.dropout = float(dropout)
        self.dropconnect = float(dropconnect)
        self.dropres = float(dropres)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(chunk_size * levels)
        self.conv_dim = self.hidden_dim
        self.conv_size = int(conv_size)
        self.levels = int(levels)
        self.chunk_size = int(chunk_size)

        self.kernel = nn.Linear(int(input_dim + 1), int(self.hidden_dim * 4 + levels * 2))
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)

        self.recurrent_kernel = nn.Linear(int(self.hidden_dim + 1), int(self.hidden_dim * 4 + levels * 2))
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(self.hidden_dim // 6), int(self.hidden_dim))
        self.nn_conv = nn.Conv1d(int(self.hidden_dim), int(self.conv_dim), int(conv_size), 1)

        if self.dropconnect > 0.0:
            self.nn_dropconnect = nn.Dropout(p=self.dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=self.dropconnect)
        if self.dropout > 0.0:
            self.nn_dropout = nn.Dropout(p=self.dropout)
        if self.dropres > 0.0:
            self.nn_dropres = nn.Dropout(p=self.dropres)

    @staticmethod
    def _cumax(x: torch.Tensor, mode: str = "l2r") -> torch.Tensor:
        if mode == "l2r":
            x = torch.softmax(x, dim=-1)
            return torch.cumsum(x, dim=-1)
        if mode == "r2l":
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        raise ValueError(f"Unsupported mode={mode!r}")

    def _step(
        self,
        inputs: torch.Tensor,
        c_last: torch.Tensor,
        h_last: torch.Tensor,
        interval: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interval = interval.unsqueeze(-1)
        x_out1 = self.kernel(torch.cat((inputs, interval), dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1))

        if self.dropconnect > 0.0:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2

        f_master_gate = self._cumax(x_out[:, : self.levels], "l2r").unsqueeze(2)
        i_master_gate = self._cumax(x_out[:, self.levels : self.levels * 2], "r2l").unsqueeze(2)

        x_out = x_out[:, self.levels * 2 :]
        x_out = x_out.reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, : self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels : self.levels * 2])
        o_gate = torch.sigmoid(x_out[:, self.levels * 2 : self.levels * 3])
        c_in = torch.tanh(x_out[:, self.levels * 3 :])

        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = (
            overlap * (f_gate * c_last + i_gate * c_in)
            + (f_master_gate - overlap) * c_last
            + (i_master_gate - overlap) * c_in
        )
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        batch_size, time_step, _ = x.size()
        device = x.device
        time = torch.ones(batch_size, time_step, device=device, dtype=x.dtype)

        c_out = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)
        h_out = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)

        tmp_h = (
            torch.zeros_like(h_out, dtype=torch.float32)
            .view(-1)
            .repeat(self.conv_size)
            .view(self.conv_size, batch_size, self.hidden_dim)
        ).to(device=device)
        tmp_dis = torch.zeros((self.conv_size, batch_size), device=device, dtype=x.dtype)

        h_list = []
        origin_h = []
        for t in range(time_step):
            out, c_out, h_out = self._step(x[:, t, :], c_out, h_out, time[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim : self.hidden_dim + self.levels], -1)
            origin_h.append(out[..., : self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., : self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)

            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)

            local_theme = torch.mean(local_h, dim=-1)
            local_theme = torch.relu(self.nn_scale(local_theme))
            local_theme = torch.sigmoid(self.nn_rescale(local_theme))

            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h_list.append(local_h)

        origin_h_t = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h_list).permute(1, 0, 2)
        if self.dropres > 0.0:
            origin_h_t = self.nn_dropres(origin_h_t)
        rnn_outputs = rnn_outputs + origin_h_t

        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1))
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)
        output = rnn_outputs.contiguous().view(batch_size, time_step, self.hidden_dim)

        last_output = last_by_lengths(output, lengths)
        return last_output, output


class StageNetModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 384,
        conv_size: int = 10,
        levels: int = 3,
        dropconnect: float = 0.3,
        dropout: float = 0.3,
        dropres: float = 0.3,
    ):
        super().__init__()
        if hidden_dim % levels != 0:
            raise ValueError("hidden_dim must be divisible by levels")
        chunk_size = hidden_dim // levels
        self.layer = StageNetLayer(
            input_dim=input_dim,
            chunk_size=chunk_size,
            conv_size=conv_size,
            levels=levels,
            dropconnect=dropconnect,
            dropout=dropout,
            dropres=dropres,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        last, _ = self.layer(x, lengths)
        return self.head(last)


class StageNetTimeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 384,
        conv_size: int = 10,
        levels: int = 3,
        dropconnect: float = 0.3,
        dropout: float = 0.3,
        dropres: float = 0.3,
    ):
        super().__init__()
        if hidden_dim % levels != 0:
            raise ValueError("hidden_dim must be divisible by levels")
        chunk_size = hidden_dim // levels
        self.layer = StageNetLayer(
            input_dim=input_dim,
            chunk_size=chunk_size,
            conv_size=conv_size,
            levels=levels,
            dropconnect=dropconnect,
            dropout=dropout,
            dropres=dropres,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        _, seq = self.layer(x, lengths)
        return self.head(seq)
