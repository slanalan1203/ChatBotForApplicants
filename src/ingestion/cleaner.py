import re
import unicodedata

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\xa0 ​]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()

def dedupe_paragraphs(text: str) -> str:
    seen = set()
    out = []
    for para in text.split("\n\n"):
        key = re.sub(r"\s+", " ", para).strip()
        if len(key) < 30 or key in seen:
            continue
        seen.add(key)
        out.append(para)
    return "\n\n".join(out)
