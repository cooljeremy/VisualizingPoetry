from typing import List, Tuple
from bert_score import score as bert_score


def bertscore_f1(cands: List[str], refs: List[str], lang: str = "zh") -> float:
    P, R, F1 = bert_score(cands, refs, lang=lang, rescale_with_baseline=True)
    return float(F1.mean().item())
