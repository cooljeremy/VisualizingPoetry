from typing import Dict, List
from pycocoevalcap.spice.spice import Spice


def spice_score(cands: Dict[str, List[str]], refs: Dict[str, List[str]]) -> float:
    scorer = Spice()
    gts = {k: v for k, v in refs.items()}
    res = {k: [v[0] if isinstance(v, list) else v] for k, v in cands.items()}
    score, _ = scorer.compute_score(gts, res)
    return float(score)


