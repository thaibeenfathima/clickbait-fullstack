import re
from typing import List, Optional
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 30
NUM_WORDS = 8000


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fit_tokenizer(texts: List[str], num_words: int = NUM_WORDS) -> Tokenizer:
    texts = [clean_text(t) for t in texts if t is not None]
    tk = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tk.fit_on_texts(texts)
    return tk


def texts_to_padded_sequences(texts: List[str], tokenizer: Tokenizer, maxlen: int = MAX_LEN):
    texts = [clean_text(t) for t in texts]
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')


# Small helper to process a single text
def preprocess_single(text: str, tokenizer: Tokenizer):
    seq = texts_to_padded_sequences([text], tokenizer)
    return seq


def preprocess_text(texts):
    """Normalize a single string or list of strings and return a list of cleaned strings.

    Args:
        texts: A single string or a list of strings.

    Returns:
        List[str]: Cleaned text strings ready for tokenization.
    """
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]
    return [clean_text(t) for t in texts]


def save_tokenizer(tokenizer, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

