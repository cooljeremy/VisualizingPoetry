from typing import List, Dict
from .pipeline import Pipeline


def run_inference(texts: List[str]) -> List[Dict]:
    pl = Pipeline()
    return pl.generate(texts)
