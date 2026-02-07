from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


from oneehr.models.utils import last_by_lengths


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = int(dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        r = torch.arange(start=1, end=number_of_logits + 1, dtype=torch.float32, device=input.device).view(1, -1)
        r = r.expand_as(zs)

        bound = 1 + r * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * r, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)
        return output.view(original_size)


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        self._causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._causal_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._causal_padding != 0:
            return y[:, :, : -self._causal_padding]
        return y


class Recalibration(nn.Module):
    def __init__(self, channel: int, reduction: int = 9, activation: str = "sigmoid"):
        super().__init__()
        self.activation = str(activation)

        self.nn_c = nn.Linear(channel, channel // reduction)
        self.nn_rescale = nn.Linear(channel // reduction, channel)
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        b, c, t = x.size()
        y_origin = x.permute(0, 2, 1).reshape(b * t, c).contiguous()
        se = torch.relu(self.nn_c(y_origin))
        y = self.nn_rescale(se).view(b, t, c).permute(0, 2, 1).contiguous()
        if self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation == "sparsemax":
            y = self.sparsemax(y)
        else:
            y = self.softmax(y)
        return x * y.expand_as(x), y.permute(0, 2, 1)


class AdaCareEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 2,
        kernel_num: int = 64,
        r_v: int = 4,
        r_c: int = 4,
        activation: str = "sigmoid",
        dropout: float = 0.5,
    ):
        super().__init__()
        if activation not in {"sigmoid", "softmax", "sparsemax"}:
            raise ValueError("activation must be one of: sigmoid, softmax, sparsemax")

        self.hidden_dim = int(hidden_dim)

        self.nn_conv1 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 1)
        self.nn_conv3 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 3)
        self.nn_conv5 = CausalConv1d(input_dim, kernel_num, kernel_size, 1, 5)
        nn.init.xavier_uniform_(self.nn_conv1.weight)
        nn.init.xavier_uniform_(self.nn_conv3.weight)
        nn.init.xavier_uniform_(self.nn_conv5.weight)

        self.nn_convse = Recalibration(3 * kernel_num, r_c, activation="sigmoid")
        self.nn_inputse = Recalibration(input_dim, r_v, activation=activation)

        self.rnn = nn.GRU(input_dim + 3 * kernel_num, hidden_dim, batch_first=True)
        self.nn_dropout = nn.Dropout(float(dropout))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        conv_input = x.permute(0, 2, 1)  # (B, D, T)
        conv_res = torch.cat(
            (self.nn_conv1(conv_input), self.nn_conv3(conv_input), self.nn_conv5(conv_input)),
            dim=1,
        )
        conv_res = self.relu(conv_res)

        convse_res, _ = self.nn_convse(conv_res)
        inputse_res, _ = self.nn_inputse(x.permute(0, 2, 1))
        concat_input = torch.cat((convse_res, inputse_res), dim=1).permute(0, 2, 1)  # (B,T,*)

        packed = nn.utils.rnn.pack_padded_sequence(
            concat_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.nn_dropout(out)


class AdaCareModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.encoder = AdaCareEncoder(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x, lengths)
        last = last_by_lengths(z, lengths)
        return self.head(last)


class AdaCareTimeModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__()
        self.encoder = AdaCareEncoder(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x, lengths)
        return self.head(z)


@dataclass(frozen=True)
class AdaCareArtifacts:
    feature_columns: list[str]
    state_dict: dict
