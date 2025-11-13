from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision import models, transforms as T


def _get_inception(device: torch.device):
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    net.eval().to(device)
    return net


def _preprocess(images: torch.Tensor) -> torch.Tensor:
    tf = T.Resize((299, 299), antialias=True)
    if images.max() > 1.0:
        images = images / 255.0
    return tf(images)


@torch.no_grad()
def inception_score_from_images(images: torch.Tensor, splits: int = 10, device: str = "cuda") -> Tuple[float, float]:
    assert images.dim() == 4 and images.size(1) == 3
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    net = _get_inception(dev)
    imgs = _preprocess(images.to(dev))
    logits = net(imgs)
    if isinstance(logits, tuple):
        logits = logits[0]
    pyx = F.softmax(logits, dim=1).cpu()
    N = pyx.size(0)
    split_scores = []
    for k in range(splits):
        part = pyx[k * (N // splits):(k + 1) * (N // splits)]
        py = part.mean(dim=0, keepdim=True)
        kl = (part * (part.add(1e-10).log() - py.add(1e-10).log())).sum(dim=1)
        split_scores.append(kl.mean().exp().item())
    import numpy as np
    return float(np.mean(split_scores)), float(np.std(split_scores))


