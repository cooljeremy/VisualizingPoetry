import torch
from typing import Dict


def export_features(seq: torch.Tensor, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"sequence": seq, "pooled": pooled}


