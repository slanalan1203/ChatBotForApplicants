import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.retriever import retrieve
from src.rag.generator import generate

def answer(query: str, k: int = 5, temperature: float = 0.3) -> dict:
    chunks = retrieve(query, k=k)
    text = generate(query, chunks, temperature=temperature)
    sources = [
        {
            "n": i + 1,
            "url": c["metadata"].get("source_url"),
            "title": c["metadata"].get("doc_title"),
        }
        for i, c in enumerate(chunks)
    ]
    return {"answer": text, "sources": sources, "chunks": chunks}

def main():
    query = " ".join(sys.argv[1:])
    if not query:
        print("usage: python -m src.rag.pipeline 'your question'")
        return
    result = answer(query)
    print(result["answer"])
    print()
    print("Источники:")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['title']} — {s['url']}")

if __name__ == "__main__":
    main()
