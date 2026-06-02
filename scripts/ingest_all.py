import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.scrapers.pdf_loader import load_pdf
from src.ingestion.scrapers.faq import fetch_faq
from src.ingestion.scrapers.hse_web import fetch_page
from src.ingestion.scrapers.vocabulary import fetch_vocabulary
from src.ingestion.cleaner import clean_text, dedupe_paragraphs
from src.ingestion.chunker import chunk_text
from src.ingestion.indexer import upsert_chunks

PDF_DIR = Path("data/raw/pdf")
KB_OUT = Path("data/processed/kb.jsonl")
PROGRAM_URLS = [
    "https://ba.hse.ru/information",
    "https://ba.hse.ru/entr",
    "https://ba.hse.ru/minkrit",
    "https://ba.hse.ru/intexam",
    "https://ba.hse.ru/tselevoe",
    "https://ba.hse.ru/price",
    "https://ba.hse.ru/discount",
    "https://ba.hse.ru/goscredit",
]

def doc_type_from_filename(name: str) -> str:
    n = name.lower()
    if "olymp" in n: return "olymp_list"
    if "rsosh" in n: return "olymp_list"
    if "bvi" in n: return "bvi"
    if n.startswith("rules_main") or n.startswith("rules_moscow"):
        return "rules"
    if "rules" in n:
        return "appendix"
    return "appendix"

DO_DEDUP = os.environ.get("DO_DEDUP", "1") == "1"

def preprocess(text: str) -> str:
    text = clean_text(text)
    if DO_DEDUP:
        text = dedupe_paragraphs(text)
    return text

def collect_pdf_chunks():
    out = []
    for path in PDF_DIR.glob("*.pdf"):
        pages = load_pdf(path)
        text = preprocess("\n\n".join(p["text"] for p in pages))
        meta = {
            "source_url": str(path),
            "source_type": "pdf",
            "doc_title": path.stem,
            "doc_type": doc_type_from_filename(path.name),
            "year": 2026,
        }
        out.extend(chunk_text(text, meta))
    return out

def collect_faq_chunks():
    items = fetch_faq()
    out = []
    for i, item in enumerate(items):
        text = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"
        meta = {
            "source_url": item["source_url"],
            "source_type": "web",
            "doc_title": "FAQ ВШЭ",
            "doc_type": "faq",
            "year": 2026,
        }
        out.append({"chunk_id": f"faq_{i}", "text": text, "char_count": len(text), **meta})
    return out

def collect_program_chunks():
    out = []
    for url in PROGRAM_URLS:
        page = fetch_page(url)
        text = preprocess(page["text"])
        meta = {
            "source_url": url,
            "source_type": "web",
            "doc_title": page["title"],
            "doc_type": "program",
            "year": 2026,
        }
        out.extend(chunk_text(text, meta))
    return out

def collect_glossary_chunks():
    return fetch_vocabulary()

def reset_collection():
    import chromadb
    client = chromadb.PersistentClient(path=os.environ.get("CHROMA_PATH", "./data/chroma"))
    try:
        client.delete_collection("hse_admission")
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["pdf", "faq", "glossary", "program"], choices=["pdf", "faq", "glossary", "program"])
    ap.add_argument("--doc-types", nargs="+", default=None,
                    help="filter chunks by doc_type after collection: rules, appendix, olymp_list, bvi, faq, glossary, program")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        reset_collection()

    chunks = []
    if "pdf" in args.sources:
        chunks.extend(collect_pdf_chunks())
    if "faq" in args.sources:
        chunks.extend(collect_faq_chunks())
    if "glossary" in args.sources:
        chunks.extend(collect_glossary_chunks())
    if "program" in args.sources:
        chunks.extend(collect_program_chunks())

    if args.doc_types:
        keep = set(args.doc_types)
        before = len(chunks)
        chunks = [c for c in chunks if c["doc_type"] in keep]
        print(f"filtered by doc_types {sorted(keep)}: {before} -> {len(chunks)}")

    KB_OUT.parent.mkdir(parents=True, exist_ok=True)
    with KB_OUT.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    upsert_chunks(chunks)
    print(f"indexed {len(chunks)} chunks")
    by_type = {}
    for c in chunks:
        by_type[c["doc_type"]] = by_type.get(c["doc_type"], 0) + 1
    print("by doc_type:", by_type)

if __name__ == "__main__":
    main()
