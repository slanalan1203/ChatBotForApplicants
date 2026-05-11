import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.retriever import retrieve

data = [json.loads(l) for l in open("data/golden_test.jsonl")]
with open("data/processed/retrieved_cache_test.jsonl", "w") as f:
    for d in data:
        chunks = retrieve(d["question"], k=5)
        f.write(json.dumps({
            "qid": d["qid"],
            "question": d["question"],
            "reference_answer": d["reference_answer"],
            "chunks": chunks,
        }, ensure_ascii=False) + "\n")
print(f"cached {len(data)} test questions")
