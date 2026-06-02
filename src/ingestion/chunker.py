import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
USE_HEADINGS = os.environ.get("USE_HEADINGS", "1") == "1"

SEPARATORS_HEADINGS = ["\n## ", "\n### ", "\n\n", ". ", " "]
SEPARATORS_FLAT = ["\n\n", ". ", " "]

def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS_HEADINGS if USE_HEADINGS else SEPARATORS_FLAT,
        length_function=len,
    )

def chunk_text(text: str, base_meta: dict) -> list[dict]:
    splitter = make_splitter()
    pieces = splitter.split_text(text)
    out = []
    for i, p in enumerate(pieces):
        rec = dict(base_meta)
        rec["chunk_id"] = f"{base_meta.get('doc_title', 'doc')}_c{i}"
        rec["text"] = p
        rec["char_count"] = len(p)
        out.append(rec)
    return out
