# evaluation/simple_metric_adapters.py
from collections import Counter




    
    
def _tok(s): return (s or "").lower().strip().split()

def exact_match(pred, refs):
    p = (pred or "").strip().lower()
    refs = [(r or "").strip().lower() for r in (refs or [])]
    return 1.0 if p in refs else 0.0

def token_f1(pred, refs):
    p = Counter(_tok(pred))
    best = 0.0
    for r in (refs or []):
        rc = Counter(_tok(r))
        inter = sum((p & rc).values())
        if inter == 0: 
            continue
        prec = inter / (sum(p.values()) + 1e-8)
        rec  = inter / (sum(rc.values()) + 1e-8)
        f1 = 2*prec*rec/(prec+rec)
        best = max(best, f1)
    return best

def _as_ids(docs):
    ids = []
    for d in docs or []:
        if isinstance(d, dict):
        
            #simple modification no additional data needed, already present
            ids.append(d.get("orriginal_id"))
            
            #also add doc_ids
            #alt_ids = d.get("alt_ids")
            #ids.extend(str(x) for x in alt_ids)
            
        else:
            ids.append(str(d))
    return [i for i in ids if i]

def _doc_overlap_at_least_k(verified_ids, gold_ids, k=1):
    return 1.0 if len(set(verified_ids) & set(gold_ids or [])) >= k else 0.0

def build_simple_metric(question, answer, ground_truth, verified_docs, gold_doc_ids):
    # normalize refs
    refs = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    refs = [str(r or "") for r in (refs or [])]
    ans  = str(answer or "")
    # correctness
    f1 = token_f1(ans, refs)
    em = exact_match(ans, refs)
    # map to judge-like raw scales:
    # relevance/equivalence raw in [-1, 2]  -> normalized = (raw+1)/3  in [0,1]
    # faithfulness raw in [-1, 1]          -> normalized = (raw+1)/2  in [0,1]
    equiv_raw = 2.0 * max(f1, em) - 1.0
    rel_raw   = 2.0 * f1 - 1.0
    # faithfulness via doc-id overlap (binary)
    
    
    #UPDATE: these are resolving to "orriginal_id" instead of numerical "id"
    v_ids = _as_ids(verified_docs)
    
   
    
    
    
    overlap_bin = _doc_overlap_at_least_k(v_ids, gold_doc_ids, k=1)
    faith_raw = (2.0 * overlap_bin) - 1.0

    relevant_score = {
        "score_relevance": rel_raw,
        "score_relevance_normalized": (rel_raw + 1.0) / 3.0,
        "score_equivalence": equiv_raw,
        "score_equivalence_normalized": (equiv_raw + 1.0) / 3.0,
        "rationals": [{"note": "simple_metrics: token_f1/em"}],
    }
    faithful_score = {
        "scor_faithfulness": faith_raw,
        "scor_faithfulness_normalized": (faith_raw + 1.0) / 2.0,
        "rationals": [{"note": "simple_metrics: doc_id_overlap>=1"}],
    }
    # a lightweight “coverage” proxy you can feed to the extractor:
    # use the same equivalence_normalized as coverage (or the F1 itself if you prefer)
    coverage = {
        "score_equivalence": equiv_raw,
        "score_equivalence_normalized": (equiv_raw + 1.0) / 3.0,
        "rationals": [{"note": "simple_metrics: coverage≈equivalence"}],
    }
    return relevant_score, faithful_score, coverage

