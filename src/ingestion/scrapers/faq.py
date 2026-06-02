import httpx
from bs4 import BeautifulSoup

FAQ_URL = "https://ba.hse.ru/faq"

def fetch_faq():
    r = httpx.get(FAQ_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    items = []
    for q_div in soup.select(".faq__question"):
        text_div = q_div.select_one(".faq__text")
        if text_div is None:
            continue
        question = text_div.get_text(" ", strip=True)
        ans = q_div.find_next_sibling()
        if ans is None:
            continue
        answer = ans.get_text(" ", strip=True)
        if not question or not answer:
            continue
        items.append({
            "question": question,
            "answer": answer,
            "source_url": FAQ_URL,
            "source_type": "web",
            "doc_type": "faq",
        })
    return items
