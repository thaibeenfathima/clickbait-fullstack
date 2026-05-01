import io
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Tuple
from .predict import annotate_headline
from .pdf_processor import extract_headlines_from_pdf

# optional streamlit cache
try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:
    def cache_data(f):
        return f


def _hash_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


@cache_data
def _cached_parse(raw_bytes: bytes, name: str):
    """Parse raw bytes into DataFrame and infer text column. Cached by Streamlit based on raw bytes hash."""
    name_lower = (name or '').lower()
    # prefer bytes streams for pandas
    try:
        if name_lower.endswith('.csv') or name_lower.endswith('.txt'):
            # assume UTF-8 text
            s = raw_bytes.decode('utf-8')
            if name_lower.endswith('.csv'):
                df = pd.read_csv(io.StringIO(s))
            else:
                rows = [r.strip() for r in s.splitlines() if r.strip()]
                df = pd.DataFrame({'headline': rows})
        elif name_lower.endswith('.json'):
            s = raw_bytes.decode('utf-8')
            df = pd.read_json(io.StringIO(s))
        elif name_lower.endswith('.xml'):
            root = ET.fromstring(raw_bytes)
            rows = []
            for child in root:
                rows.append({c.tag: (c.text or '').strip() for c in child})
            df = pd.DataFrame(rows)
        elif name_lower.endswith('.pdf'):
            # extract lines from PDF bytes
            from io import BytesIO
            lines = extract_headlines_from_pdf(BytesIO(raw_bytes))
            df = pd.DataFrame({'headline': lines})
        elif name_lower.endswith('.xlsx') or name_lower.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            # try CSV first, then fallback to reading as lines
            try:
                s = raw_bytes.decode('utf-8')
                df = pd.read_csv(io.StringIO(s))
            except Exception:
                rows = [r.strip() for r in s.splitlines() if r.strip()]
                df = pd.DataFrame({'headline': rows})
    except Exception:
        df = pd.DataFrame()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # heuristics to guess text column
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['headline', 'title', 'text', 'content'])]
    text_col = candidates[0] if candidates else (df.columns[0] if len(df.columns) > 0 else None)
    return df, text_col


def load_file_to_df(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Accepts an uploaded file (Streamlit UploadedFile) or path-like and returns (df, inferred_text_column_name).

    Supports CSV, JSON, XLSX, XML, TXT and PDF (text lines) - PDF should be pre-extracted by `pdf_processor` when used via Streamlit.

    Uses content hashing to cache parsing results (reduces repeated parsing on re-uploads).
    """
    name = getattr(uploaded_file, 'name', None) or str(uploaded_file)
    # read bytes in one shot so we can compute hash and cache parse
    raw = None
    try:
        raw = uploaded_file.read()
    except Exception:
        # uploaded_file might be a path-like object
        try:
            with open(str(uploaded_file), 'rb') as f:
                raw = f.read()
        except Exception:
            raw = b''
    if raw is None:
        raw = b''
    # ensure bytes
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    file_hash = _hash_bytes(raw)
    df, text_col = _cached_parse(raw, name)
    return df, text_col, file_hash


def _process_batch_core(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()
    if text_column is None or text_column not in df.columns:
        raise ValueError('Text column not found')
    df[text_column] = df[text_column].astype(str)
    annotations = [annotate_headline(h) for h in df[text_column].tolist()]
    ann_df = pd.DataFrame(annotations)
    # Avoid duplicate column names (e.g., 'headline' from annotations and original)
    if 'headline' in ann_df.columns:
        ann_df = ann_df.drop(columns=['headline'])
    # Prefix prediction columns to make them unique and avoid conflicts with original columns
    rename_map = {
        'clickbait_label': 'pred_clickbait_label',
        'confidence': 'pred_confidence',
        'sentiment': 'pred_sentiment',
        'sentiment_score': 'pred_sentiment_score'
    }
    ann_df = ann_df.rename(columns=rename_map)
    # preserve original order and merge
    out = pd.concat([df.reset_index(drop=True), ann_df.reset_index(drop=True)], axis=1)
    # Ensure columns are unique (pyarrow/parquet and downstream tools require unique names)
    if out.columns.duplicated().any():
        # As a fallback, make duplicate column names unique by appending suffixes
        cols = []
        seen = {}
        for c in out.columns:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        out.columns = cols
    return out


def process_batch(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Process batch and cache results based on df contents and selected column (if running in Streamlit).

    Uses a content-derived hash for stable cache keys (avoids reprocessing identical uploads).
    """
    try:
        # serialize DataFrame for stable hashing
        df_json = df.to_json(orient='split')
        df_hash = _hash_bytes(df_json.encode('utf-8') + (text_column or '').encode('utf-8'))

        @cache_data
        def _cached_by_hash(hash_key, df_json_inner, col):
            # df_json_inner is stored to avoid re-reading the original df in outer scope
            df_inner = pd.read_json(io.StringIO(df_json_inner), orient='split')
            return _process_batch_core(df_inner, col)

        return _cached_by_hash(df_hash, df_json, text_column)
    except Exception:
        return _process_batch_core(df, text_column)
