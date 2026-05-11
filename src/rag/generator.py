import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import ollama

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

SYSTEM = (
    "Ты — ассистент абитуриента ВШЭ. Отвечай только на основе предоставленного контекста. "
    'Если в контексте нет ответа — скажи "В моих источниках этого нет, рекомендую обратиться в приёмную комиссию". '
    "Цитируй источники в формате [1], [2]. Отвечай кратко и по делу, на русском языке."
)

def build_prompt(query: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    return f"Контекст:\n{ctx}\n\nВопрос: {query}\n\nОтвет:"

def generate(query: str, chunks: list[dict], temperature: float = 0.3) -> str:
    prompt = build_prompt(query, chunks)
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return resp["message"]["content"]
