import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from rouge_score import rouge_scorer
from src.rag.agent import agent_answer
from src.rag.pipeline import answer as baseline_answer

GOLDEN = Path("data/golden_dev.jsonl")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", choices=["multi_hop", "hard", "all"], default="multi_hop")
    args = ap.parse_args()

    data = [json.loads(l) for l in GOLDEN.open()]
    if args.filter == "multi_hop":
        data = [d for d in data if d.get("needs_multi_hop")]
    elif args.filter == "hard":
        data = [d for d in data if d.get("difficulty") == "hard"]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    results = []

    for i, d in enumerate(data):
        # baseline
        t0 = time.time()
        base = baseline_answer(d["question"], k=5, temperature=0.0)
        base_lat = time.time() - t0
        base_r = scorer.score(d["reference_answer"], base["answer"])["rougeL"].fmeasure

        # agentic
        ag = agent_answer(d["question"])
        ag_r = scorer.score(d["reference_answer"], ag["answer"])["rougeL"].fmeasure

        results.append({
            "qid": d["qid"],
            "question": d["question"],
            "reference": d["reference_answer"],
            "baseline_answer": base["answer"],
            "baseline_rouge_l": base_r,
            "baseline_latency": base_lat,
            "agent_answer": ag["answer"],
            "agent_rouge_l": ag_r,
            "agent_steps": ag["steps"],
            "agent_latency": ag["latency"],
            "agent_trace": ag["trace"],
        })
        print(f"{i+1}/{len(data)} {d['qid']:>10} | base ROUGE={base_r:.2f} t={base_lat:.1f}s | agent ROUGE={ag_r:.2f} steps={ag['steps']} t={ag['latency']:.1f}s", flush=True)

    n = len(results)
    summary = {
        "n": n,
        "filter": args.filter,
        "baseline_avg_rouge_l": sum(r["baseline_rouge_l"] for r in results) / n,
        "agent_avg_rouge_l": sum(r["agent_rouge_l"] for r in results) / n,
        "baseline_avg_latency": sum(r["baseline_latency"] for r in results) / n,
        "agent_avg_latency": sum(r["agent_latency"] for r in results) / n,
        "agent_avg_steps": sum(r["agent_steps"] for r in results) / n,
        "agent_exhausted": sum(1 for r in results if r["agent_trace"] and any("exhausted" in str(t) for t in r["agent_trace"])),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
