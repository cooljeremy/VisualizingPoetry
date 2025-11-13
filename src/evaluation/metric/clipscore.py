from typing import List
import torch
import open_clip
from PIL import Image
from torchvision import transforms as T


def _load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=dev)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, dev


@torch.no_grad()
def clipscore(texts: List[str], images: List[Image.Image], model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda") -> float:
    model, preprocess, tokenizer, dev = _load_clip(model_name, pretrained, device)
    toks = tokenizer(texts).to(dev)
    ims = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(dev)
    txt_feat = model.encode_text(toks)
    img_feat = model.encode_image(ims)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    sim = (txt_feat * img_feat).sum(dim=-1)
    return float(sim.mean().item())


