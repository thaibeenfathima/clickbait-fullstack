from pathlib import Path
import sys
p=Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0,str(p))

from src.explainability import integrated_gradients

text = "You won't believe what happened to this cat"
print('Running IG on sample text:')
print(text)
res = integrated_gradients(text, steps=12)
print('IG results:', res)
