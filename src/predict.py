import os
import pickle
from typing import List, Tuple
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Handle transformers import with fallback for Windows torch DLL issues
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Transformers library not available ({str(e)[:80]}...). Some features will be limited.")
    TRANSFORMERS_AVAILABLE = False
    # Dummy pipeline function for fallback
    def pipeline(*args, **kwargs):
        raise ImportError("Transformers not available. Update torch or use Python 3.11")

# optional streamlit caching decorators
try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except Exception:
    # fallback no-op decorators when streamlit isn't available
    def cache_resource(f):
        return f
    def cache_data(f):
        return f

from src.preprocess import preprocess_text, load_tokenizer, NUM_WORDS, MAX_LEN
from src.train_model import train_dummy_model, DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH

MODEL_DIR = "models"
MODEL_PATH = DEFAULT_MODEL_PATH
TOKENIZER_PATH = DEFAULT_TOKENIZER_PATH

tokenizer = None
model = None
sentiment_analyzer = None
# lazy headline generator (transformer-based)
headline_generator = None
headline_generator_model = None
GENERATOR_DEFAULT_MODEL = 'distilgpt2'


@cache_resource
def _load_model_and_tokenizer_cached(model_path: str = MODEL_PATH, tokenizer_path: str = TOKENIZER_PATH):
    """Load (or quick-train) and return (model, tokenizer). Cached to avoid repeated loads."""
    tk = load_tokenizer(tokenizer_path)
    if tk is not None and os.path.exists(model_path):
        m = load_model(model_path, compile=False)
        return m, tk
    # fallback: quick train
    m, tk = train_dummy_model()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    m.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tk, f)
    return m, tk


def _ensure_model_and_tokenizer():
    """Ensure globals `model` and `tokenizer` are set (uses cached loader)."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    m, tk = _load_model_and_tokenizer_cached()
    model = m
    tokenizer = tk


@cache_resource
def _load_sentiment_pipeline():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        print(f"Warning: Could not load sentiment analyzer: {e}")
        return None


def _ensure_sentiment():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = _load_sentiment_pipeline()


def predict_proba(text: str) -> float:
    """Return probability that text is clickbait."""
    _ensure_model_and_tokenizer()
    if tokenizer is None or model is None:
        raise RuntimeError("Model/tokenizer not available")
    texts = preprocess_text(text)
    seq = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seq, maxlen=MAX_LEN)
    p = float(model.predict(X, verbose=0)[0][0])
    return p


# ------------------ Transformer-based headline generator ------------------

@cache_resource
def _get_generator_pipeline(model_name: str = GENERATOR_DEFAULT_MODEL):
    """Return a cached transformer pipeline for the given model_name.

    Uses text2text generation for t5-like models, and text-generation for GPT-like models.
    """
    if not TRANSFORMERS_AVAILABLE:
        print(f"Warning: Transformers not available. Headline generation disabled.")
        return None
    task = 'text2text-generation' if 't5' in model_name.lower() else 'text-generation'
    try:
        return pipeline(task, model=model_name)
    except Exception as e:
        print(f"Warning: Could not load generator pipeline: {e}")
        return None


def _ensure_generator(model_name: str = GENERATOR_DEFAULT_MODEL):
    global headline_generator, headline_generator_model
    if headline_generator is not None and headline_generator_model == model_name:
        return
    gen = _get_generator_pipeline(model_name)
    headline_generator = gen
    headline_generator_model = model_name if gen is not None else None


def is_serious_headline(text: str) -> bool:
    """Return True if headline contains keywords typical for serious/news events."""
    kws = {'dies','died','killed','kill','attack','violence','migrant','police','court','arrest','war','disaster','death','dead','shooting','murder','suicide','fatal','crash','fire','collapse'}
    txt = text.lower()
    return any(k in txt for k in kws)


def suggest_headlines(text: str, n: int = 4, news_style: bool = False) -> List[str]:
    """Return template-based headline suggestions. If `news_style` is True, use
    neutral, journalistic templates instead of clickbait-styled ones."""
    news_templates = [
        "Breaking: {}",
        "Update: {}",
        "{} — reports say",
        "{} — officials confirm",
        "{}: Latest developments"
    ]
    clickbait_templates = [
        "Breaking: {}",
        "You won't believe: {}",
        "Top story: {}",
        "How to {} in 5 steps",
        "The surprising reason {}"
    ]
    templates = news_templates if news_style else clickbait_templates
    out = []
    for t in templates[:n]:
        out.append(t.format(text if len(text) < 80 else text[:80].rstrip()))
    return out


def generate_headlines(text: str, n: int = 5, max_words: int = 15, model_name: str = GENERATOR_DEFAULT_MODEL, sentiment: str = None, rewrite_only: bool = False, allow_ai: bool = True, force_news_style: bool = False) -> List[str]:
    """Generate headline variations using a transformer.

    Behavior:
      - Detects serious/news headlines and can force news-style templates.
      - AI generation is attempted only when `allow_ai` is True, headline is short (< max_words),
        and the content is non-serious (not force_news_style).
      - If AI fails quality checks, falls back silently to templates (news or clickbait depending on `force_news_style`).

    Args:
        text: original headline
        n: preferred number of final suggestions
        max_words: maximum allowed words in a generated suggestion
        model_name: HF model name
        sentiment: optional sentiment instruction
        rewrite_only: if True, use rewrite prompt and stricter filters
        allow_ai: if False, do not call the HF pipeline and use templates
        force_news_style: if True, use neutral journalistic templates for fallback and avoid clickbait templates

    Returns:
        List[str]: 3-5 unique, headline-style suggestions
    """
    # quick guard
    if not text or not text.strip():
        return []

    import re
    from typing import Set

    # small stopword set for keyword extraction
    STOPWORDS = {
        'the','and','a','an','in','on','at','for','to','of','is','are','was','were','be','by','with','from','that','this','it','as','your','you'
    }
    # casual/chat words to reject
    CASUAL_WORDS = {'dude','song','video','link','lol','omg','haha','wtf','lmao','subscribe','youtube','click here','watch','see this','trailer','episode','vlog'}

    def extract_keywords(s: str) -> Set[str]:
        toks = re.findall(r"\w+", s.lower())
        return set([t for t in toks if len(t) > 2 and t not in STOPWORDS])

    keywords = extract_keywords(text)

    is_serious = force_news_style or is_serious_headline(text)

    # Decide whether to attempt AI generation
    words_in_text = re.findall(r"\w+", text)
    can_use_ai = allow_ai and (len(words_in_text) < max_words) and (not is_serious)

    # If AI is not allowed, immediately return templates (news or clickbait)
    if not can_use_ai:
        return suggest_headlines(text, n=min(5, n), news_style=is_serious)

    # build rewrite-only prompt when requested
    instr = ''
    if sentiment and not rewrite_only:
        instr = f' Make it sound {sentiment.lower()}.'

    if rewrite_only:
        # strict rewrite prompt (do not add new topics)
        if 't5' in model_name.lower():
            prompt = f"Rewrite the following headline into a short news or clickbait headline (max {max_words} words). Do not add new topics: \"{text}\""
        else:
            prompt = f"Rewrite the following headline into a short news or clickbait headline (max {max_words} words). Do not add new topics: \"{text}\"\nOutput:"
    else:
        if 't5' in model_name.lower():
            prompt = f'paraphrase: {text} Make it a short catchy clickbait headline (<= {max_words} words).{instr}'
        else:
            prompt = f'Headline: "{text}"\nShort catchy clickbait headline:{instr} '

    try:
        _ensure_generator(model_name=model_name)
        if headline_generator is None:
            raise RuntimeError('Generator not available')
        generator_pipeline = headline_generator

        # request more candidates to filter down; cap num_return_sequences
        num_requests = min(10, max(8, n * 3))
        outs = generator_pipeline(prompt, max_new_tokens=40, do_sample=True, top_p=0.92, temperature=0.7, num_return_sequences=num_requests)

        candidates = []
        seen = set()

        def contains_keyword(candidate: str) -> bool:
            if not keywords:
                return True
            toks = set(re.findall(r"\w+", candidate.lower()))
            return len(toks & keywords) > 0

        for o in outs:
            raw = o.get('generated_text', '')
            # remove prompt prefix if present
            if raw.startswith(prompt):
                generated = raw[len(prompt):].strip()
            else:
                generated = raw.replace(text, '').strip()
            # take first line/sentence
            gen = generated.split('\n')[0].split('. ')[0].strip()
            # strip quotes and surrounding punctuation
            gen = gen.strip(' "\'').strip()
            # basic checks
            if not gen or not any(c.isalpha() for c in gen):
                continue
            # drop outputs with replacement chars or heavy non-ascii
            if '�' in gen:
                continue
            ascii_chars = sum(1 for c in gen if ord(c) < 128)
            if ascii_chars / max(1, len(gen)) < 0.75:
                continue
            # drop casual/chat words
            low = gen.lower()
            if any(cw in low for cw in CASUAL_WORDS):
                continue
            # disallow obviously clickbaity phrases when original is serious
            if is_serious and any(phr in low for phr in ["how to", "you won't believe", "you won't believe", "you'll never guess"]):
                continue
            # enforce word limits
            words = re.findall(r"\w+", gen)
            if len(words) < 2:
                continue
            if len(words) > max_words:
                gen = ' '.join(words[:max_words])
                words = re.findall(r"\w+", gen)
            # remove trailing punctuation
            gen = gen.rstrip('.,!;:')
            # reject outputs that end awkwardly or with stop tokens (likely not headline-style)
            BAD_END_TOKENS = {'and','but','or','too','was','were','is','are','be','been','has','have','had','of','in','for','with','to','the','a','an','that','which','who','where','when','why','how'}
            if words and words[-1].lower() in BAD_END_TOKENS:
                continue
            # skip if identical or contains the full original headline (we want rewritten variants)
            key = gen.lower().strip()
            if key == text.lower().strip():
                continue
            if text.lower().strip() in key:
                continue
            # require at least one keyword
            if not contains_keyword(gen):
                continue
            # skip repeats
            if key in seen:
                continue
            # avoid conversational phrasing
            if re.search(r"\b(i|i'm|i\bam|we|we're|im|you know)\b", key):
                continue
            # reject narrative-like starts and weak phrasing
            if re.match(r"^(this is|it is|there is)\b", key):
                continue
            if 'a little' in key or 'little bit' in key or 'a bit' in key:
                continue
            # avoid relative clauses for longer candidates
            if ('that' in key or 'which' in key) and len(words) > 7:
                continue
            # limit absolute length to avoid long sentences
            if len(gen) > 120:
                continue
            seen.add(key)
            candidates.append(gen)

        # reduce to 3-5 final unique headlines; silently fallback to templates if too few
        final = []
        for c in candidates:
            if c.lower() not in [x.lower() for x in final]:
                final.append(c)
            if len(final) >= min(5, n):
                break
        if len(final) < 3:
            return suggest_headlines(text, n=min(5, n), news_style=is_serious)
        return final[:min(5, n)]
    except Exception:
        return suggest_headlines(text, n=min(5, n), news_style=is_serious)


def generate_sentiment_variations(text: str, n_per_sentiment: int = 3) -> dict:
    """Generate sentiment-guided headline variations. Returns dict with keys 'POSITIVE','NEGATIVE','NEUTRAL'."""
    outs = {}
    for s in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        outs[s] = generate_headlines(text, n=n_per_sentiment, sentiment=s)
    return outs


def predict_clickbait(text: str) -> Tuple[str, float]:
    p = predict_proba(text)
    label = "Clickbait" if p >= 0.5 else "Non-Clickbait"
    return label, p


def predict_batch(headlines: List[str]) -> List[Tuple[str, float]]:
    _ensure_model_and_tokenizer()
    texts = preprocess_text(headlines)
    seq = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seq, maxlen=MAX_LEN)
    probs = model.predict(X, verbose=0).flatten().tolist()
    return [("Clickbait" if p >= 0.5 else "Non-Clickbait", float(p)) for p in probs]


def get_sentiment(text: str) -> Tuple[str, float]:
    _ensure_sentiment()
    if sentiment_analyzer is None:
        return "NEUTRAL", 0.0
    res = sentiment_analyzer(text)[0]
    label = res.get('label', 'NEUTRAL').upper()
    score = float(res.get('score', 0.0))
    return label, score




def get_word_importance(text: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """Return list of (word, importance) where importance is drop in prob when word removed."""
    base = predict_proba(text)
    words = text.split()
    scores = []
    for i, w in enumerate(words):
        masked = words[:i] + words[i+1:]
        masked_text = " ".join(masked)
        try:
            p = predict_proba(masked_text) if masked_text.strip() else 0.0
        except Exception:
            p = base
        scores.append((w, base - p))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores_sorted[:top_n]


def annotate_headline(headline: str) -> dict:
    label, conf = predict_clickbait(headline)
    sent_label, sent_score = get_sentiment(headline)
    important = get_word_importance(headline, top_n=3)
    suggestions = suggest_headlines(headline, n=3, news_style=is_serious_headline(headline))
    return {
        'headline': headline,
        'clickbait_label': label,
        'confidence': float(conf),
        'sentiment': sent_label,
        'sentiment_score': float(sent_score),
        'important_words': [w for w, _ in important],
        'word_scores': [s for _, s in important],
        'suggestions': suggestions
    }
