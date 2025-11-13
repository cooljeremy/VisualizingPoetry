import torch


def tensor_interaction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,bj->bij", a, b).flatten(1)


