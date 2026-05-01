import pdfplumber


def extract_headlines_from_pdf(pdf_file) -> list:
    """Extract text lines from a PDF file-like object and return non-empty lines."""
    lines = []
    try:
        # pdf_file may be a path or file-like
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if not t:
                    continue
                for line in t.split('\n'):
                    text = line.strip()
                    if text:
                        lines.append(text)
    except Exception:
        return []
    return lines
