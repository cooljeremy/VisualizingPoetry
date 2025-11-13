from typing import Dict, List, Optional
from PIL import Image
import torch
from .metrics.inception_score import inception_score_from_images
from .metrics.fid import fid_from_images
from .metrics.clipscore import clipscore
from .metrics.cider import cider_score
from .metrics.bertscore import bertscore_f1
from .metrics.spice import spice_score


def evaluate_all():
    return {
        "is": 0.0,
        "fid": 0.0,
        "clipscore": 0.0,
        "cider": 0.0,
        "bertscore": 0.0,
        "spice": 0.0,
    }


def evaluate_metrics(gen_images: Optional[torch.Tensor], real_images: Optional[torch.Tensor], gen_captions: Dict[str, List[str]], ref_captions: Dict[str, List[str]], clip_texts: List[str], clip_images: List[Image.Image], lang: str = "zh", device: str = "cuda") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if gen_images is not None:
        is_mean, _ = inception_score_from_images(gen_images, device=device)
        out["is"] = is_mean
    if gen_images is not None and real_images is not None:
        out["fid"] = fid_from_images(real_images, gen_images, device=device)
    if clip_texts and clip_images:
        out["clipscore"] = clipscore(clip_texts, clip_images, device=device)
    if gen_captions and ref_captions:
        out["cider"] = cider_score(gen_captions, ref_captions)
        can_list = [gen_captions[k][0] for k in gen_captions.keys()]
        ref_list = [ref_captions[k][0] for k in ref_captions.keys()]
        out["bertscore"] = bertscore_f1(can_list, ref_list, lang=lang)
        out["spice"] = spice_score(gen_captions, ref_captions)
    return out
