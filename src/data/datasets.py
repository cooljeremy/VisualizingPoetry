from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class PoeticVisDataset(Dataset):
    def __init__(self, root: str, split_file: str, max_length: int = 128) -> None:
        self.root = Path(root)
        self.items = []
        split_path = self.root / split_file
        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.items.append(json.loads(line))

        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        poem = item.get("poem", "")
        image_path = item.get("image", "")
        return {"poem": poem, "image_path": str((self.root / image_path).as_posix())}


