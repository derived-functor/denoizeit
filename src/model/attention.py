"""Implementation of bottleneck attention"""

import torch
from torch import nn


class BottleneckAttention(nn.Module):
    """Bottleneck attention for U-Net"""

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 512 (complex), F, T)
        """
        b, c, f, t = x.shape

        # (B, 512, F, T, 2)
        x = torch.view_as_real(x)

        # (B, F * T, C * 2)
        x = x.permute(0, 2, 3, 1, 4).reshape(b, f * t, c * 2)

        x = self.ln(x)
        x = self.transformer(x)

        # (B, 512, F, T)
        x = x.reshape(b, f, t, c, 2)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = torch.view_as_complex(x)

        return x
