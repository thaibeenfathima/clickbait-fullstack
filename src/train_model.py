import os
import sys
import pickle
from typing import Tuple
import numpy as np
from .preprocess import fit_tokenizer, texts_to_padded_sequences, NUM_WORDS, MAX_LEN
from .attention import AttentionLayer as SimpleAttention
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout

MODEL_MAXLEN = MAX_LEN
DEFAULT_MODEL_PATH = os.path.join("models", "clickbait_model.h5")
DEFAULT_TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")


def build_model(vocab_size: int = NUM_WORDS, embedding_dim: int = 64, maxlen: int = MODEL_MAXLEN) -> Model:
    inp = Input(shape=(maxlen,), name='input')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, mask_zero=True)(inp)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = SimpleAttention()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_on_data(texts, labels, epochs: int = 5, save_model_path: str = DEFAULT_MODEL_PATH, save_tokenizer_path: str = DEFAULT_TOKENIZER_PATH):
    tokenizer = fit_tokenizer(texts)
    X = texts_to_padded_sequences(texts, tokenizer)
    y = np.array(labels)

    model = build_model(vocab_size=NUM_WORDS)
    model.fit(X, y, epochs=epochs, batch_size=8, verbose=1)

    # ensure models dir exists
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    model.save(save_model_path)

    # save tokenizer (use pickle)
    with open(save_tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    return model, tokenizer


def train_dummy_model(sample_texts=None, sample_labels=None, epochs: int = 3):
    # Create a lightweight synthetic dataset when no dataset is provided.
    if sample_texts is None or sample_labels is None:
        sample_texts = [
            "This one trick will change your life",
            "10 reasons why cats are better than dogs",
            "You won't believe what happened next",
            "Scientists discover new species",
            "Local man helps neighbor with groceries",
            "New study reveals health benefits of walking",
            "How to save money this year",
            "Breaking: celebrity announces new album",
            "The simple way to improve your sleep",
            "Experts explain how to build a birdhouse"
        ]
        sample_labels = [1,1,1,0,0,0,0,1,0,0]

    tokenizer = fit_tokenizer(sample_texts + sample_texts)
    X = texts_to_padded_sequences(sample_texts, tokenizer)
    y = np.array(sample_labels)

    model = build_model(vocab_size=NUM_WORDS)
    model.fit(X, y, epochs=epochs, batch_size=4, verbose=0)
    return model, tokenizer


# Allow execution as script
if __name__ == '__main__':
    # allow relative import when run as script
    try:
        # Data path relative to project root
        DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clickbait_data.csv')
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        texts = df['headline'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        print(f"Training on {len(texts)} samples...")
        train_on_data(texts, labels, epochs=5)
        print(f"Training complete. Saved model to {DEFAULT_MODEL_PATH} and tokenizer to {DEFAULT_TOKENIZER_PATH}")
    except Exception as e:
        print('Failed to run training script:', e)
        sys.exit(1)

