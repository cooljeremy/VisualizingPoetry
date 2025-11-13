from typing import Callable, Dict, List, Tuple
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from .text_relevance import TextEmbedder
from .topic_model import topic_vector
from .clip_similarity import encode_with_openclip
from .combiner import textual_similarity, total_score


def s_textual(poems: List[str], prompts: List[str], alpha: float = 0.5, model_name: str = "bert-base-chinese", device: str = "cuda") -> float:
    te = TextEmbedder(model_name=model_name, device=device)
    e_poem = te.encode(poems)
    e_prompt = te.encode(prompts)
    s_text = torch.nn.functional.cosine_similarity(e_poem, e_prompt).mean().item()
    tp = np.array(topic_vector(poems, k=10))
    tr = np.array(topic_vector(prompts, k=10))
    tp_n = tp / (np.linalg.norm(tp, axis=1, keepdims=True) + 1e-8)
    tr_n = tr / (np.linalg.norm(tr, axis=1, keepdims=True) + 1e-8)
    s_topic = float((tp_n * tr_n).sum(axis=1).mean())
    return float(alpha * s_text + (1.0 - alpha) * s_topic)


def s_image_text(texts: List[str], images: List[Image.Image], model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda") -> float:
    img_feat, txt_feat = encode_with_openclip(texts, images, model_name=model_name, pretrained=pretrained, device=device)
    img = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    sim = (img * txt).sum(dim=-1).mean().item()
    return float(sim)


def evaluate_and_maybe_regenerate(poems: List[str], prompts: List[str], images: List[Image.Image], alpha: float = 0.5, beta: float = 0.5, thresh: float = 0.8, regenerate_fn: Callable[[List[str]], Tuple[List[str], List[Image.Image]]] = None, max_iter: int = 2, device: str = "cuda") -> Dict[str, float]:
    st = s_textual(poems, prompts, alpha=alpha, device=device)
    si = s_image_text(prompts, images, device=device)
    s_total = total_score(st, si, beta=beta)
    it = 0
    cur_prompts = prompts
    cur_images = images
    while s_total < thresh and regenerate_fn is not None and it < max_iter:
        cur_prompts, cur_images = regenerate_fn(cur_prompts)
        st = s_textual(poems, cur_prompts, alpha=alpha, device=device)
        si = s_image_text(cur_prompts, cur_images, device=device)
        s_total = total_score(st, si, beta=beta)
        it += 1
    return {"S_textual": st, "S_image_text": si, "S_total": s_total}


