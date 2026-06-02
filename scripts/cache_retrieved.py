import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from src.rag.retriever import retrieve

GOLDEN = Path("data/golden_dev.jsonl")
OUT = Path("data/processed/retrieved_cache.jsonl")
TOP_K = 5

def main():
    data = [json.loads(l) for l in GOLDEN.open()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for d in data:
            chunks = retrieve(d["question"], k=TOP_K)
            rec = {
                "qid": d["qid"],
                "question": d["question"],
                "reference_answer": d["reference_answer"],
                "chunks": chunks,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"cached {len(data)} questions x top-{TOP_K} chunks to {OUT}")

if __name__ == "__main__":
    main()
