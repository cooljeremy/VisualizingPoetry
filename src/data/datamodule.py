from typing import Optional
from torch.utils.data import DataLoader
from .datasets import PoeticVisDataset
from .transforms import build_transforms


def build_dataloaders(root: str, train_split: str, val_split: str, test_split: str, image_size: int, batch_size: int, num_workers: int, max_length: int):
    train_set = PoeticVisDataset(root, train_split, max_length)
    val_set = PoeticVisDataset(root, val_split, max_length)
    test_set = PoeticVisDataset(root, test_split, max_length)
    _, _ = build_transforms(image_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


