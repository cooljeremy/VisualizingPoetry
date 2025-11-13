import torch
from torch import nn


class PromptDecoder(nn.Module):
    def __init__(self, dim: int, layers: int, vocab_size: int = 30522) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, batch_first=True)
        self.dec = nn.TransformerDecoder(encoder_layer, num_layers=layers)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, h: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        tgt = self.embed(tgt_ids)
        out = self.dec(tgt, h.unsqueeze(1))
        return self.proj(out)


