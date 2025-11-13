import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    def __init__(self, in_dim: int, emo_classes: int = 8, img_classes: int = 16, rhe_classes: int = 20) -> None:
        super().__init__()
        self.emo = nn.Linear(in_dim, emo_classes)
        self.img = nn.Linear(in_dim, img_classes)
        self.rhe = nn.Linear(in_dim, rhe_classes)

    def forward(self, x: torch.Tensor):
        return {"emo": self.emo(x), "img": self.img(x), "rhe": self.rhe(x)}


