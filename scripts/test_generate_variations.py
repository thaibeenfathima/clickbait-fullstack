from pathlib import Path
import sys
p=Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0,str(p))

from src.predict import generate_headlines, suggest_headlines

text = "You won't believe what happened to this cat"
print('Template suggestions (deterministic):')
print(suggest_headlines(text, n=5))
print('\nAI-generated variations (run 1):')
print(generate_headlines(text, n=5, model_name='distilgpt2'))
print('\nAI-generated variations (run 2):')
print(generate_headlines(text, n=5, model_name='distilgpt2'))
print('\nAI-generated variations (run 3):')
print(generate_headlines(text, n=5, model_name='distilgpt2'))
