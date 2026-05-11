def hit_at_k(retrieved: list, relevant: list, k: int) -> float:
    return float(any(r in set(relevant) for r in retrieved[:k]))

def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant)
    found = sum(1 for r in retrieved[:k] if r in rel)
    return found / len(rel)

def reciprocal_rank(retrieved: list, relevant: list) -> float:
    rel = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0
