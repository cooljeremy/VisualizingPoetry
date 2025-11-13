from typing import Dict, List
import torch
import torch.nn.functional as F


def combine(scores: Dict[str, float], w_text: float = 0.5, w_image: float = 0.5) -> float:
    return w_text * scores.get("text", 0.0) + w_image * scores.get("image", 0.0)


def textual_similarity(emb_poem: torch.Tensor, emb_prompt: torch.Tensor) -> float:
    s = F.cosine_similarity(emb_poem, emb_prompt).mean().item()
    return float(s)


def total_score(s_textual: float, s_imtxt: float, beta: float = 0.5) -> float:
    return float(beta * s_textual + (1.0 - beta) * s_imtxt)


