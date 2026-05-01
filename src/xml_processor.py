import io
import pandas as pd
from lxml import etree


def xml_to_dataframe(raw_bytes: bytes) -> pd.DataFrame:
    try:
        root = etree.fromstring(raw_bytes)
        rows = []
        # If XML contains records under root children
        for child in root:
            row = {}
            for elem in child.iterchildren():
                row[elem.tag] = (elem.text or '').strip()
            if row:
                rows.append(row)
        if rows:
            return pd.DataFrame(rows)
        # fallback: collect text nodes
        texts = [etree.tostring(e, method='text', encoding='utf-8').decode('utf-8').strip() for e in root.iter()]
        texts = [t for t in texts if t]
        return pd.DataFrame({'text': texts})
    except Exception:
        return pd.DataFrame()
