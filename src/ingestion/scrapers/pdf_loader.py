import fitz
from pathlib import Path

def load_pdf(path: Path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({
            "page_num": i + 1,
            "text": page.get_text(),
            "source_url": str(path),
            "source_type": "pdf",
            "doc_title": path.stem,
        })
    return pages
