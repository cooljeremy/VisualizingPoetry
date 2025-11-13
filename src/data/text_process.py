from typing import List


def tokenize(texts: List[str], max_length: int) -> List[List[int]]:
    return [[0] * min(len(t), max_length) for t in texts]


