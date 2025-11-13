import torch
from torch import nn


class FeatureFusion(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, emo: torch.Tensor, img: torch.Tensor, rhe: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emo, img, rhe], dim=-1)
        return self.proj(x)
