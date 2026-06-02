import os
import re
import time
import ollama
from src.rag.retriever import retrieve

MODEL = os.environ.get("OLLAMA_MODEL", "mistral:latest")
MAX_STEPS = 3
TOP_K = 5

SYSTEM = (
    "Ты — агент-помощник абитуриента ВШЭ. Решаешь вопросы пошагово через ReAct-петлю. "
    "Доступен один инструмент: search_kb(query: str, doc_type: str|None). "
    "doc_type может быть: rules, faq, glossary, olymp_list, program, appendix, или null. "
    "Формат каждого шага:\n"
    "Thought: краткое рассуждение что нужно искать\n"
    "Action: search_kb(\"запрос\", \"doc_type или null\")\n"
    "Observation: <результат подставит система>\n\n"
    "Когда готов финальный ответ, выведи:\n"
    "Final Answer: <ответ на русском, с указанием источников>\n\n"
    f"Максимум {MAX_STEPS} шагов. После последнего search_kb ОБЯЗАТЕЛЬНО выдай Final Answer."
)

ACTION_RE = re.compile(r'Action:\s*search_kb\(\s*"([^"]+)"\s*(?:,\s*"?([^",\)]+)"?)?\s*\)', re.IGNORECASE)
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)

def parse_action(text: str):
    m = ACTION_RE.search(text)
    if not m:
        return None
    query = m.group(1).strip()
    doc_type = m.group(2)
    if doc_type and doc_type.lower() in ("null", "none", ""):
        doc_type = None
    return query, doc_type

def format_observation(chunks: list[dict]) -> str:
    if not chunks:
        return "(ничего не найдено)"
    return "\n".join(
        f"[{i+1}] doc_type={c['metadata'].get('doc_type','?')} | {c['text'][:300]}"
        for i, c in enumerate(chunks)
    )

def agent_answer(query: str) -> dict:
    history = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]
    trace = []
    t0 = time.time()

    for step in range(MAX_STEPS):
        resp = ollama.chat(model=MODEL, messages=history, options={"temperature": 0.0, "num_predict": 600})
        text = resp["message"]["content"]
        trace.append({"step": step + 1, "model_output": text})

        final = FINAL_RE.search(text)
        if final:
            answer = final.group(1).strip()
            return {
                "answer": answer,
                "trace": trace,
                "steps": step + 1,
                "latency": time.time() - t0,
            }

        action = parse_action(text)
        if action is None:
            trace.append({"step": step + 1, "error": "no action parsed"})
            return {
                "answer": text.strip(),
                "trace": trace,
                "steps": step + 1,
                "latency": time.time() - t0,
            }

        sub_query, doc_type = action
        chunks = retrieve(sub_query, k=TOP_K)
        if doc_type:
            chunks = [c for c in chunks if c["metadata"].get("doc_type") == doc_type]
        obs = format_observation(chunks)
        trace[-1]["action"] = {"query": sub_query, "doc_type": doc_type, "n_results": len(chunks)}
        trace[-1]["observation_preview"] = obs[:300]

        history.append({"role": "assistant", "content": text})
        history.append({"role": "user", "content": f"Observation:\n{obs}\n\nПродолжай."})

    return {
        "answer": text.strip() if text else "(не сформирован)",
        "trace": trace,
        "steps": MAX_STEPS,
        "latency": time.time() - t0,
        "exhausted": True,
    }
