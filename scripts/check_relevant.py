import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import get_collection

GOLDEN = Path("data/golden_set.jsonl")
SAMPLE_QIDS = ["kt1_01", "kt1_25", "kt1_08", "new_05", "new_07"]

def main():
    data = {d["qid"]: d for d in (json.loads(l) for l in GOLDEN.open())}
    coll = get_collection()
    for qid in SAMPLE_QIDS:
        if qid not in data:
            continue
        d = data[qid]
        print(f"\n{'='*80}\n{qid}: {d['question']}\nREF: {d['reference_answer'][:200]}")
        print(f"category: {d['category']} | difficulty: {d['difficulty']}")
        print(f"relevant_chunk_ids: {d['relevant_chunk_ids']}")
        if not d["relevant_chunk_ids"]:
            print("(none)")
            continue
        got = coll.get(ids=d["relevant_chunk_ids"], include=["documents", "metadatas"])
        for i, (cid, doc, meta) in enumerate(zip(got["ids"], got["documents"], got["metadatas"])):
            print(f"\n  [{i+1}] {cid} (doc_type={meta.get('doc_type')})")
            print(f"      {doc[:300]}")

if __name__ == "__main__":
    main()
