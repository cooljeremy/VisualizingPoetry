import torch
from torch import nn


class MultiHeadAttn(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, k, v, need_weights=False)
        return out


