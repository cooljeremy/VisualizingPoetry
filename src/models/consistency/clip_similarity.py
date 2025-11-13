import torch
from typing import List
from PIL import Image
import open_clip


def clip_similarity(img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
    img = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return (img * txt).sum(dim=-1)


@torch.no_grad()
def encode_with_openclip(texts: List[str], images: List[Image.Image], model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=dev)
    tokenizer = open_clip.get_tokenizer(model_name)
    toks = tokenizer(texts).to(dev)
    ims = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(dev)
    txt_feat = model.encode_text(toks)
    img_feat = model.encode_image(ims)
    return img_feat, txt_feat

