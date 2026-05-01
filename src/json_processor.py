import io
import json
import pandas as pd


def json_to_dataframe(raw_bytes: bytes) -> pd.DataFrame:
    try:
        s = raw_bytes.decode('utf-8')
    except Exception:
        try:
            s = str(raw_bytes)
        except Exception:
            return pd.DataFrame()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return pd.json_normalize(obj)
        if isinstance(obj, dict):
            # Try to normalize nested dict
            return pd.json_normalize(obj)
    except Exception:
        # try line-delimited JSON
        try:
            rows = [json.loads(line) for line in s.splitlines() if line.strip()]
            return pd.json_normalize(rows)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()
