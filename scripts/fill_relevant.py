import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import ollama
from src.rag.retriever import retrieve

JUDGE_MODEL = "qwen2.5:7b-instruct"
GOLDEN = Path("data/golden_set.jsonl")
OUT = Path("data/golden_set_filled.jsonl")
TOP_K = 5
TEXT_LIMIT = 300

PROMPT = """Вопрос: {question}
Эталон: {ref}

Фрагменты:
{candidates}

Верни JSON-массив номеров фрагментов, релевантных вопросу. Например: [1, 3] или []. Только массив, без пояснений."""

def judge(question, ref, candidates):
    cand_text = "\n".join(
        f"[{i+1}] {c['text'][:TEXT_LIMIT]}"
        for i, c in enumerate(candidates)
    )
    prompt = PROMPT.format(question=question, ref=ref, candidates=cand_text)
    resp = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 50},
    )
    text = resp["message"]["content"]
    m = re.search(r"\[[\d,\s]*\]", text)
    if m is None:
        return []
    indices = json.loads(m.group())
    return [candidates[i-1]["id"] for i in indices if isinstance(i, int) and 1 <= i <= len(candidates)]

def main():
    data = [json.loads(l) for l in GOLDEN.open()]
    n = len(data)
    no_relevant = 0
    for i, d in enumerate(data):
        candidates = retrieve(d["question"], k=TOP_K)
        relevant = judge(d["question"], d["reference_answer"], candidates)
        d["relevant_chunk_ids"] = relevant
        if not relevant:
            no_relevant += 1
        print(f"{i+1}/{n} {d['qid']:>15}: {len(relevant)}/{len(candidates)} | {d['question'][:55]}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    avg = sum(len(d["relevant_chunk_ids"]) for d in data) / n
    print(f"\nwrote {OUT}", flush=True)
    print(f"avg relevant per q: {avg:.2f}", flush=True)
    print(f"questions with 0 relevant: {no_relevant}/{n}", flush=True)

if __name__ == "__main__":
    main()
