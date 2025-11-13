from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from .logger import Logger
from .checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model: nn.Module, optimizer, scheduler, step_fn: Callable, device: str, out_dir: str, log_interval: int = 50) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.device = device
        self.logger = Logger(out_dir)
        self.out_dir = out_dir
        self.log_interval = log_interval

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        total = 0.0
        for i, batch in enumerate(loader):
            self.optimizer.zero_grad()
            loss = self.step_fn(self.model, batch, self.device)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            if (i + 1) % self.log_interval == 0:
                self.logger.log(f"epoch={epoch} step={i+1} loss={loss.item():.4f}")
        return total / max(len(loader), 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        for e in range(1, epochs + 1):
            tr_loss = self.train_epoch(train_loader, e)
            self.scheduler.step()
            save_checkpoint({"epoch": e, "state_dict": self.model.state_dict()}, self.out_dir, f"epoch{e}")
            self.logger.log(f"epoch_end={e} train_loss={tr_loss:.4f}")
