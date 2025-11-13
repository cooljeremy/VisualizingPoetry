from torch.optim import AdamW


def build_optimizer(params, lr: float, weight_decay: float):
    return AdamW(params, lr=lr, weight_decay=weight_decay)


