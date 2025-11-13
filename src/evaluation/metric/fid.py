from typing import Tuple
import numpy as np
import torch
from torchvision import models, transforms as T


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _get_inception_feature_extractor(device: torch.device):
    net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
    net.fc = Identity()
    net.eval().to(device)
    return net


def _preprocess(images: torch.Tensor) -> torch.Tensor:
    tf = T.Resize((299, 299), antialias=True)
    if images.max() > 1.0:
        images = images / 255.0
    return tf(images)


@torch.no_grad()
def inception_activations(images: torch.Tensor, device: str = "cuda") -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    net = _get_inception_feature_extractor(dev)
    imgs = _preprocess(images.to(dev))
    feats = net(imgs)
    if isinstance(feats, tuple):
        feats = feats[0]
    return feats.cpu().numpy()


def _stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def fid_from_stats(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(diff.dot(diff) + tr)


def fid_from_images(real_images: torch.Tensor, gen_images: torch.Tensor, device: str = "cuda") -> float:
    r = inception_activations(real_images, device)
    g = inception_activations(gen_images, device)
    mu1, sigma1 = _stats(r)
    mu2, sigma2 = _stats(g)
    return fid_from_stats(mu1, sigma1, mu2, sigma2)


