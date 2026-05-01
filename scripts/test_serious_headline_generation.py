import sys
from pathlib import Path
p = Path(__file__).resolve().parents[1]
if str(p) not in sys.path:
    sys.path.insert(0, str(p))

from src.predict import generate_headlines, suggest_headlines

text = "Four killed in apartment fire downtown"
outs = generate_headlines(text, n=5, max_words=15, rewrite_only=True)
print('Generated (serious):', outs)

templates = suggest_headlines(text, n=5, news_style=True)
assert outs == templates, 'Serious headlines must use news-style templates as fallback or be news-style'
for o in outs:
    low = o.lower()
    assert 'how to' not in low and "you won't" not in low and 'you wont' not in low, 'No clickbait phrases in serious suggestions'
print('Serious headline generation test passed ✅')
