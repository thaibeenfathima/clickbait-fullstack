from pathlib import Path
import sys
p=Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0,str(p))
from src.batch_processor import load_file_to_df, process_batch
import pandas as pd

# create sample CSV bytes
csv = "headline\nYou won't believe what happened to this cat\nAnother headline here\n"
from io import BytesIO
f = BytesIO(csv.encode('utf-8'))
# Simulate uploaded file with name attribute
class DummyFile:
    def __init__(self, buf, name='sample.csv'):
        self._buf = buf
        self.name = name
    def read(self):
        self._buf.seek(0)
        return self._buf.read()

uploaded = DummyFile(BytesIO(csv.encode('utf-8')))
df, text_col, file_hash = load_file_to_df(uploaded)
print('parsed df columns:', df.columns.tolist())
print('text_col:', text_col)
print('file_hash:', file_hash)

# Run process_batch twice and ensure consistent results
res1 = process_batch(df, text_col)
res2 = process_batch(df, text_col)
print('res1 cols:', res1.columns.tolist())
print('res2 cols:', res2.columns.tolist())
assert res1.equals(res2)
print('Batch processing consistent (cached)')
