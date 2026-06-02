import httpx
from bs4 import BeautifulSoup

VOCAB_URL = "https://ba.hse.ru/vocabulary"

ABBREVS = {
    "индивидуальные достижения": "ИД",
    "контрольные цифры приема": "КЦП",
    "контрольные цифры приёма": "КЦП",
    "дополнительные вступительные испытания": "ДВИ",
    "без вступительных испытаний": "БВИ",
    "ограниченные возможности здоровья": "ОВЗ",
    "всероссийская олимпиада школьников": "ВсОШ",
    "единый государственный экзамен": "ЕГЭ",
    "среднее профессиональное образование": "СПО",
    "целевое обучение": "ЦО",
}

def find_abbrev(term: str) -> str:
    t = term.lower()
    for full, abbr in ABBREVS.items():
        if full in t:
            return abbr
    return ""

def fetch_vocabulary():
    r = httpx.get(VOCAB_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    fronts = soup.select(".card-front")
    backs = soup.select(".card-back")
    out = []
    for i, (front, back) in enumerate(zip(fronts, backs)):
        term = front.get_text(" ", strip=True)
        defn = back.get_text(" ", strip=True)
        if not term or not defn:
            continue
        abbr = find_abbrev(term)
        head = f"{abbr} ({term})" if abbr and abbr not in term else term
        text = f"{head}: {defn}"
        out.append({
            "chunk_id": f"vocab_{i}",
            "text": text,
            "char_count": len(text),
            "source_url": VOCAB_URL,
            "source_type": "web",
            "doc_title": "Словарь абитуриента ВШЭ",
            "doc_type": "glossary",
            "section": term,
            "year": 2026,
        })
    return out
