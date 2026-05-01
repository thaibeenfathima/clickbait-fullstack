from pathlib import Path
import sys
p=Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0,str(p))
from src.batch_processor import process_batch
import pandas as pd

df=pd.DataFrame({'headline':["You won't believe what happened to this cat"]})
out=process_batch(df,'headline')
print('COLUMNS:', out.columns.tolist())
print('ROW0:', out.iloc[0].to_dict())
