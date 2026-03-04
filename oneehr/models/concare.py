from __future__ import annotations

import torch
from torch import nn


def _make_pad_mask(lengths: torch.Tensor, t: int) -> torch.Tensor:
    # True means ignore.
    return torch.arange(t, device=lengths.device)[None, :] >= lengths[:, None]


class FinalAttentionQKV(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_type: str = "add", dropout: float = 0.1):
        super().__init__()
        self.attention_type = str(attention_type)
        self.hidden_dim = int(hidden_dim)

        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(p=float(dropout))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.Wh = nn.Parameter(torch.randn(2 * input_dim, hidden_dim))
        self.Wa = nn.Parameter(torch.randn(hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        bsz, t, _ = x.size()
        q = self.W_q(x[:, -1, :])  # (B, H)
        k = self.W_k(x)  # (B, T, H)
        v = self.W_v(x)  # (B, T, H)

        if self.attention_type == "add":
            q2 = q.view(bsz, 1, self.hidden_dim)
            h = self.tanh(q2 + k)
            e = self.W_out(h).view(bsz, t)
        elif self.attention_type == "mul":
            q2 = q.view(bsz, self.hidden_dim, 1)
            e = torch.matmul(k, q2).squeeze(-1)
        elif self.attention_type == "concat":
            q2 = q.unsqueeze(1).repeat(1, t, 1)
            c = torch.cat((q2, k), dim=-1)
            h = self.tanh(torch.matmul(c, self.Wh))
            e = (torch.matmul(h, self.Wa) + self.ba).view(bsz, t)
        else:
            raise ValueError(f"Unsupported attention_type={self.attention_type!r}")

        a = self.softmax(e)
        a = self.dropout(a)
        ctx = torch.matmul(a.unsqueeze(1), v).squeeze(1)
        return ctx, a


class _ConCareTransformer(nn.Module):
    """Shared transformer backbone for ConCare patient and time modes."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 static_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.static_dim = int(static_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=float(dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.static_proj = nn.Identity() if self.static_dim == 0 else nn.Linear(static_dim, hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        t = h.shape[1]
        pad_mask = _make_pad_mask(lengths, t)
        return self.encoder(h, src_key_padding_mask=pad_mask)


class ConCareEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 static_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.backbone = _ConCareTransformer(input_dim, hidden_dim, num_heads, static_dim=static_dim, dropout=dropout)
        self.final_attn = FinalAttentionQKV(hidden_dim, hidden_dim, attention_type="add", dropout=dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x, lengths)
        ctx, attn = self.final_attn(z)
        return ctx, attn


class ConCareModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 static_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.static_dim = int(static_dim)
        self.encoder = ConCareEncoder(input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                                      static_dim=static_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim * (2 if static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        ctx, _ = self.encoder(x, lengths)
        if static is not None and self.static_dim > 0:
            s = self.encoder.backbone.static_proj(static)
            ctx = torch.cat([ctx, s], dim=-1)
        return self.head(ctx)


class ConCareTimeModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 static_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.static_dim = int(static_dim)
        self.backbone = _ConCareTransformer(input_dim, hidden_dim, num_heads, static_dim=static_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim * (2 if static_dim else 1), 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        z = self.backbone(x, lengths)
        if static is not None and self.static_dim > 0:
            s = self.backbone.static_proj(static).unsqueeze(1).expand(-1, z.shape[1], -1)
            z = torch.cat([z, s], dim=-1)
        return self.head(z)
