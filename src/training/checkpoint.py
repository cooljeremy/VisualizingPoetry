from pathlib import Path
import torch


def save_checkpoint(state, out_dir: str, name: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.pt"
    torch.save(state, path.as_posix())
    return path.as_posix()


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")


