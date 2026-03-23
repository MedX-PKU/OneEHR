"""AdaCare model for patient-level and time-level prediction.

Paper: Liantao Ma et al. Adacare: Explainable clinical health status
representation learning via scale-adaptive feature extraction and
recalibration. AAAI 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oneehr.models.recurrent import last_by_lengths


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.size()
        x = x.view(-1, x.size(self.dim))
        dim = 1
        n_logits = x.size(dim)

        x = x - x.max(dim=dim, keepdim=True)[0]
        zs = x.sort(dim=dim, descending=True)[0]
        rng = torch.arange(1, n_logits + 1, dtype=x.dtype, device=x.device).view(1, -1)

        bound = 1 + rng * zs
        cumsum = torch.cumsum(zs, dim)
        is_gt = (bound > cumsum).to(x.dtype)
        k = (is_gt * rng).max(dim, keepdim=True)[0]

        taus = (torch.sum(is_gt * zs, dim, keepdim=True) - 1) / k
        out = torch.clamp(x - taus, min=0)
        return out.view(original_size)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self._pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=self._pad, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self._pad != 0:
            return out[:, :, : -self._pad]
        return out


class Recalibration(nn.Module):
    def __init__(self, channel: int, reduction: int = 9, activation: str = "sigmoid"):
        super().__init__()
        self.fc = nn.Linear(channel, channel // reduction)
        self.rescale = nn.Linear(channel // reduction, channel)
        self.activation = activation
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, t = x.size()
        y = x.permute(0, 2, 1).reshape(b * t, c)
        y = torch.relu(self.fc(y))
        y = self.rescale(y).view(b, t, c).permute(0, 2, 1)
        if self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation == "sparsemax":
            y = self.sparsemax(y)
        else:
            y = torch.softmax(y, dim=1)
        return x * y, y.permute(0, 2, 1)


class AdaCareLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 2,
        kernel_num: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = CausalConv1d(input_dim, kernel_num, kernel_size, dilation=1)
        self.conv3 = CausalConv1d(input_dim, kernel_num, kernel_size, dilation=3)
        self.conv5 = CausalConv1d(input_dim, kernel_num, kernel_size, dilation=5)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

        self.conv_recal = Recalibration(3 * kernel_num, max(1, (3 * kernel_num) // 4), "sigmoid")
        self.input_recal = Recalibration(input_dim, max(1, input_dim // 4), "sigmoid")

        self.rnn = nn.GRU(input_dim + 3 * kernel_num, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        conv_in = x.permute(0, 2, 1)
        c1 = self.conv1(conv_in)
        c3 = self.conv3(conv_in)
        c5 = self.conv5(conv_in)
        conv_cat = torch.relu(torch.cat([c1, c3, c5], dim=1))

        conv_out, _ = self.conv_recal(conv_cat)
        inp_out, _ = self.input_recal(x.permute(0, 2, 1))
        combined = torch.cat([conv_out, inp_out], dim=1).permute(0, 2, 1)

        packed = nn.utils.rnn.pack_padded_sequence(
            combined,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output


class AdaCareModel(nn.Module):
    """Patient-level AdaCare."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        kernel_size: int = 2,
        kernel_num: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layer = AdaCareLayer(input_dim, hidden_dim, kernel_size, kernel_num, dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.layer(x, lengths)
        last = last_by_lengths(out, lengths)
        return self.head(self.dropout(last))


class AdaCareTimeModel(nn.Module):
    """Time-level AdaCare."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 1,
        kernel_size: int = 2,
        kernel_num: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layer = AdaCareLayer(input_dim, hidden_dim, kernel_size, kernel_num, dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.layer(x, lengths)
        return self.head(self.dropout(out))
