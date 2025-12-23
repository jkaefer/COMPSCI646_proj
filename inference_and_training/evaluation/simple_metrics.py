# evaluation/simple_metrics.py

from collections import Counter
from typing import List, Sequence, Tuple, Iterable

def _tokenize(s: str) -> List[str]:
    return (s or "").lower().strip().split()

def exact_match(pred: str, references: Sequence[str]) -> float:
    p = (pred or "").strip().lower()
    refs = [(r or "").strip().lower() for r in (references or [])]
    return 1.0 if p in refs else 0.0  # :contentReference[oaicite:2]{index=2}

def token_f1(pred: str, references: Sequence[str]) -> float:
    p_toks = _tokenize(pred)
    if not references:
        return 0.0
    pc = Counter(p_toks)
    best = 0.0
    for ref in references:
        rc = Counter(_tokenize(ref))
        overlap = sum((pc & rc).values())
        if overlap == 0:
            continue
        prec = overlap / (sum(pc.values()) + 1e-8)
        rec  = overlap / (sum(rc.values()) + 1e-8)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best  # :contentReference[oaicite:3]{index=3}

def is_faithful_by_docs(verified_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int = 1) -> float:
    s_ver = set(filter(None, verified_doc_ids or []))
    s_gold = set(filter(None, gold_doc_ids or []))
    return 1.0 if len(s_ver & s_gold) >= k else 0.0  # :contentReference[oaicite:4]{index=4}

def grounding_prf(verified_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Tuple[float, float, float]:
    """Precision/Recall/F1 on doc-id overlap."""
    s_ver = set(filter(None, verified_doc_ids or []))
    s_gold = set(filter(None, gold_doc_ids or []))
    if not s_ver and not s_gold:
        return (1.0, 1.0, 1.0)
    if not s_ver or not s_gold:
        return (0.0, 0.0, 0.0)
    inter = len(s_ver & s_gold)
    prec = inter / len(s_ver)
    rec  = inter / len(s_gold)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return (prec, rec, f1)

