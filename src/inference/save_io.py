from pathlib import Path
from typing import List, Dict
import json


def save_results(items: List[Dict], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "results.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path.as_posix()


