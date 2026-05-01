import requests
from bs4 import BeautifulSoup


def extract_headlines_from_url(url: str, timeout: int = 8) -> list:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DeClickify/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(r.text, 'html.parser')
    headings = []
    for tag in ['h1', 'h2', 'h3']:
        for h in soup.find_all(tag):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)
    # dedupe while preserving order
    seen = set()
    unique = []
    for h in headings:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique
