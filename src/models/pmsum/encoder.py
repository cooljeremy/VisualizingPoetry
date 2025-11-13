from typing import List, Tuple
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-chinese") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, texts: List[str], max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        outputs = self.encoder(**enc)
        return outputs.last_hidden_state, outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)

