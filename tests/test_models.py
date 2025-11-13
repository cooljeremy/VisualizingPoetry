import torch
from poetic_vis.models.fusion.fuse import FeatureFusion


def test_fusion_forward():
    m = FeatureFusion(1536, 512)
    o = m(torch.randn(2, 512), torch.randn(2, 512), torch.randn(2, 512))
    assert o.shape == (2, 512)


