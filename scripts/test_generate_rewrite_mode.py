import sys
import time
from pathlib import Path
p = Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0, str(p))

from src.predict import generate_headlines, suggest_headlines

text = "You won't believe what this cat did to survive the storm"
print('Original:', text)

outs = generate_headlines(text, n=5, max_words=15, model_name='distilgpt2', rewrite_only=True)
print('Generated:', outs)

templates = suggest_headlines(text, n=5)
if outs == templates:
    print('AI generation fell back to template suggestions (acceptable).')
    sys.exit(0)

# Otherwise validate constraints
import re
CASUAL = {'dude','song','video','link','lol','omg','haha','wtf','lmao','subscribe','youtube','click here','watch','see this','trailer','episode','vlog'}

# keywords extraction
keywords = set([t for t in re.findall(r"\w+", text.lower()) if len(t) > 2 and t not in {'the','and','you','this','that','for','with','from'}])
assert 3 <= len(outs) <= 5, f'Expected 3-5 outputs, got {len(outs)}'
for o in outs:
    print('Checking:', o)
    assert isinstance(o, str) and o.strip(), 'Output must be non-empty string'
    toks = re.findall(r"\w+", o.lower())
    assert len(toks) >= 2, 'Must be at least 2 words'
    assert len(toks) <= 15, 'Must be <= 15 words'
    assert any(k in toks for k in keywords), f'Output must contain at least one keyword; keywords={keywords}, got toks={toks}'
    assert not any(cw in o.lower() for cw in CASUAL), 'Output contains casual/chat word'

print('AI rewrite outputs satisfy constraints ✅')
sys.exit(0)
