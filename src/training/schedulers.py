from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(optimizer, epochs: int):
    return CosineAnnealingLR(optimizer, T_max=epochs)


