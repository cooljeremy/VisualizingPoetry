from typing import Dict, List
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider


def cider_score(cands: Dict[str, List[str]], refs: Dict[str, List[str]]) -> float:
    scorer = Cider()
    gts = {k: v for k, v in refs.items()}
    res = {k: [v[0] if isinstance(v, list) else v] for k, v in cands.items()}
    score, _ = scorer.compute_score(gts, res)
    return float(score)


