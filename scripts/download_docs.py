import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

PDF_DIR = Path("data/raw/pdf")
BASE = "https://ba.hse.ru"

DOCS = [
    ("rules_main", "/mirror/pubs/share/1134464639"),
    ("rules_moscow", "/mirror/pubs/share/1120607478"),
    ("rules_indiv_achievements_moscow", "/mirror/pubs/share/1134464667"),
    ("rules_ovz", "/mirror/pubs/share/1120607698"),
    ("rules_subject_mapping", "/mirror/pubs/share/1120612500"),
    ("olymp_rules", "/mirror/pubs/share/1120660204"),
    ("olymp_moscow_rights", "/mirror/pubs/share/1120660254"),
    ("appeal_regulation", "/mirror/pubs/share/1139355308.pdf"),
    ("competitions_results_regulation", "/mirror/pubs/share/1068595063.pdf"),
    ("paid_contract_rules", "/mirror/pubs/share/952389098.pdf"),
    ("rules_spo_mapping", "/mirror/pubs/share/1120612598.pdf"),
]

def download(name: str, url: str):
    full = url if url.startswith("http") else BASE + url
    out = PDF_DIR / f"{name}.pdf"
    if out.exists():
        print(f"skip {name} (exists)")
        return
    r = httpx.get(full, timeout=60, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"got {name}: {len(r.content)} bytes")

def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS:
        download(name, url)

if __name__ == "__main__":
    main()
