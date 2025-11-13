import torch
from torch import nn


class MTLLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        le = self.ce(preds["emo"], targets["emo"])
        li = self.ce(preds["img"], targets["img"])
        lr = self.ce(preds["rhe"], targets["rhe"])
        return le + li + lr


