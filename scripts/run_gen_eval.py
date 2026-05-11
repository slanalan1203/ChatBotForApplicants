import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import ollama
from rouge_score import rouge_scorer

CACHE = Path("data/processed/retrieved_cache.jsonl")

PROMPTS = {
    "simple": (
        "Ты — ассистент абитуриента ВШЭ. Отвечай только на основе контекста.\n\n"
        "Контекст:\n{ctx}\n\nВопрос: {q}\n\nОтвет:"
    ),
    "citing": (
        "Ты — ассистент абитуриента ВШЭ. Отвечай только на основе контекста. Цитируй источники в формате [N].\n\n"
        "Контекст:\n{ctx}\n\nВопрос: {q}\n\nОтвет (с цитатами):"
    ),
    "fewshot": (
        "Ты — ассистент абитуриента ВШЭ. Отвечай кратко, по делу, ссылаясь на источники.\n\n"
        "Пример 1:\n"
        "Контекст:\n[1] БВИ — без вступительных испытаний.\n"
        "Вопрос: Что такое БВИ?\n"
        "Ответ: БВИ означает «без вступительных испытаний» [1].\n\n"
        "Пример 2:\n"
        "Контекст:\n[1] Документы подаются до 25 июля.\n"
        "Вопрос: Когда дедлайн?\n"
        "Ответ: До 25 июля [1].\n\n"
        "Теперь твой вопрос:\n"
        "Контекст:\n{ctx}\n\nВопрос: {q}\n\nОтвет:"
    ),
}

def build_prompt(template_key: str, question: str, chunks: list) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    return PROMPTS[template_key].format(ctx=ctx, q=question)

def generate(model: str, prompt: str, temperature: float, max_tokens: int = 400) -> str:
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return resp["response"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="simple")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cache = [json.loads(l) for l in CACHE.open()]
    if args.limit:
        cache = cache[:args.limit]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    answers = []
    rouge_l_scores = []
    latencies = []

    for i, item in enumerate(cache):
        prompt = build_prompt(args.prompt, item["question"], item["chunks"])
        t0 = time.time()
        ans = generate(args.model, prompt, args.temperature)
        dt = time.time() - t0
        scores = scorer.score(item["reference_answer"], ans)
        rl = scores["rougeL"].fmeasure
        rouge_l_scores.append(rl)
        latencies.append(dt)
        answers.append({
            "qid": item["qid"],
            "question": item["question"],
            "reference": item["reference_answer"],
            "answer": ans,
            "rouge_l": rl,
            "latency": dt,
        })
        print(f"{i+1}/{len(cache)} {item['qid']:>15} rouge-L={rl:.3f} t={dt:.1f}s", flush=True)

    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "temperature": args.temperature,
        "n": len(cache),
        "avg_rouge_l": sum(rouge_l_scores) / len(rouge_l_scores),
        "avg_latency": sum(latencies) / len(latencies),
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "answers": answers}, f, ensure_ascii=False, indent=2)

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
