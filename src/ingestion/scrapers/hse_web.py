import httpx
import trafilatura

def fetch_page(url: str):
    r = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = trafilatura.extract(r.text, url=url, include_comments=False, include_tables=True)
    meta = trafilatura.extract_metadata(r.text)
    return {
        "url": url,
        "html": r.text,
        "text": text or "",
        "title": meta.title if meta else "",
    }
