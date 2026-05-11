import json
from pathlib import Path
from src.rag.retriever import retrieve as _retrieve
from src.eval.metrics import hit_at_k, recall_at_k, reciprocal_rank

def retrieve_ids(query: str, k: int = 10) -> list[str]:
    return [r["id"] for r in _retrieve(query, k=k)]

def evaluate(golden_path: Path, k_list=(1, 3, 5, 10)):
    data = [json.loads(l) for l in golden_path.open()]
    scored = [d for d in data if d["relevant_chunk_ids"]]
    results = {f"hit@{k}": [] for k in k_list}
    results.update({f"recall@{k}": [] for k in k_list})
    results["mrr"] = []
    per_q = []

    for d in scored:
        ret = retrieve_ids(d["question"], k=max(k_list))
        for k in k_list:
            results[f"hit@{k}"].append(hit_at_k(ret, d["relevant_chunk_ids"], k))
            results[f"recall@{k}"].append(recall_at_k(ret, d["relevant_chunk_ids"], k))
        results["mrr"].append(reciprocal_rank(ret, d["relevant_chunk_ids"]))
        per_q.append({"qid": d["qid"], "category": d["category"], "retrieved": ret[:5], "relevant": d["relevant_chunk_ids"]})

    summary = {m: sum(vs) / len(vs) for m, vs in results.items()}
    summary["n_scored"] = len(scored)
    summary["n_skipped_no_relevant"] = len(data) - len(scored)
    return summary, per_q
