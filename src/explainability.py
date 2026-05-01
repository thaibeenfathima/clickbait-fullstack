from typing import List, Tuple
import html
import numpy as np

# optional streamlit cache
try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:
    def cache_data(f):
        return f

from src.predict import get_word_importance
from src import preprocess, predict
import tensorflow as tf


def highlight_important_words(headline: str, top_n: int = 3, use_ig: bool = False) -> str:
    """Return a markdown string with top important words emphasized (bold) and the rest plain.

    If use_ig=True, use Integrated Gradients to compute top words; otherwise fall back to leave-one-out.
    """
    if not headline:
        return ""
    if use_ig:
        try:
            ig = integrated_gradients(headline, steps=20)
            important = set([w for w, _ in ig[:top_n]])
        except Exception:
            important = set([w for w, _ in get_word_importance(headline, top_n=top_n)])
    else:
        important = set([w for w, _ in get_word_importance(headline, top_n=top_n)])
    parts = []
    for w in headline.split():
        w_escaped = html.escape(w)
        if w in important:
            parts.append(f"**{w_escaped}**")
        else:
            parts.append(w_escaped)
    return " ".join(parts)


def top_words_from_batch(headlines: List[str], top_k: int = 20) -> List[tuple]:
    """Simple frequency of words across headlines - used for histogram."""
    from collections import Counter
    words = []
    for h in headlines:
        for w in h.split():
            w = w.lower().strip('.,!?:;"\'')
            if len(w) > 2:
                words.append(w)
    c = Counter(words)
    return c.most_common(top_k)


@cache_data
def integrated_gradients(text: str, steps: int = 20) -> List[Tuple[str, float]]:
    """Compute Integrated Gradients (IG) attribution scores for each non-padding token in the input text.

    Returns a list of (word, score) sorted by score descending.

    Notes:
    - Uses the loaded model and tokenizer from `src.predict` (ensures they are loaded).
    - Uses the embedding layer outputs for IG computations.
    - Caches results using Streamlit's `st.cache_data` when available.
    """
    if not text or not text.strip():
        return []

    # ensure model/tokenizer
    predict._ensure_model_and_tokenizer()
    model = predict.model
    tokenizer = predict.tokenizer
    if model is None or tokenizer is None:
        return []

    # prepare input sequence and embeddings
    seq = preprocess.texts_to_padded_sequences([text], tokenizer)
    seq_tf = tf.constant(seq, dtype=tf.int32)

    # find embedding layer
    emb_idx = None
    for i, l in enumerate(model.layers):
        if isinstance(l, tf.keras.layers.Embedding):
            emb_idx = i
            emb_layer = l
            break
    if emb_idx is None:
        return []

    # create suffix model that maps embedding outputs to final prediction
    embedding_dim = int(emb_layer.output_shape[-1])
    maxlen = int(model.input_shape[1])
    emb_input = tf.keras.Input(shape=(maxlen, embedding_dim))
    x = emb_input
    for layer in model.layers[emb_idx + 1:]:
        x = layer(x)
    suffix_model = tf.keras.Model(inputs=emb_input, outputs=x)

    # get baseline and input embeddings
    baseline_tokens = tf.zeros_like(seq_tf)
    baseline_emb = emb_layer(baseline_tokens)
    input_emb = emb_layer(seq_tf)

    # integrate gradients
    alphas = np.linspace(0.0, 1.0, num=steps, endpoint=True)
    total_grads = np.zeros_like(input_emb.numpy(), dtype=float)

    for a in alphas:
        emb = baseline_emb + (input_emb - baseline_emb) * float(a)
        with tf.GradientTape() as tape:
            tape.watch(emb)
            pred = suffix_model(emb)
            # pred shape (batch, 1)
            out = tf.squeeze(pred, axis=-1)
        grads = tape.gradient(out, emb)
        if grads is not None:
            total_grads += grads.numpy()

    avg_grads = total_grads / len(alphas)
    attributions = (input_emb.numpy() - baseline_emb.numpy()) * avg_grads
    token_importances = np.sum(attributions, axis=-1).squeeze().tolist()  # shape (maxlen,)

    # map token ids back to words and keep only non-padding tokens
    seq_list = seq[0].tolist()
    index_word = getattr(tokenizer, 'index_word', None) or {v: k for k, v in tokenizer.word_index.items()}
    results = []
    for tok, score in zip(seq_list, token_importances):
        if tok == 0:
            break
        word = index_word.get(tok, f"tok_{tok}")
        results.append((word, float(score)))

    # normalize scores (optional): make positive and sort
    # keep as raw attribution values so sign gives directionality
    results_sorted = sorted(results, key=lambda x: abs(x[1]), reverse=True)
    return results_sorted
