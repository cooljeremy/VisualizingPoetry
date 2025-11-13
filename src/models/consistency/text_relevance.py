from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b)


class TextEmbedder:
    def __init__(self, model_name: str = "bert-base-chinese", device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        batch = self.tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
        out = self.enc(**batch)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)


