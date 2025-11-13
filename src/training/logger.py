from pathlib import Path


class Logger:
    def __init__(self, out_dir: str) -> None:
        self.path = Path(out_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / "train.log"

    def log(self, msg: str) -> None:
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


