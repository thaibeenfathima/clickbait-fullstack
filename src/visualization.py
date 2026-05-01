import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd


def _get_col(df: pd.DataFrame, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def plot_clickbait_distribution(df: pd.DataFrame):
    col = _get_col(df, ['pred_clickbait_label', 'clickbait_label'])
    if col is None:
        raise KeyError('No clickbait label column found (expected pred_clickbait_label or clickbait_label)')
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(5,4))
    counts.plot(kind='bar', color=['#FF8C00', '#2E8B57'], ax=ax)
    ax.set_title('Clickbait vs Non-Clickbait')
    ax.set_ylabel('Count')
    fig.tight_layout()
    return fig


def plot_sentiment_distribution(df: pd.DataFrame):
    col = _get_col(df, ['pred_sentiment', 'sentiment'])
    if col is None:
        raise KeyError('No sentiment column found (expected pred_sentiment or sentiment)')
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(5,4))
    counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title('Sentiment Distribution')
    fig.tight_layout()
    return fig


def plot_confidence_histogram(df: pd.DataFrame):
    col = _get_col(df, ['pred_confidence', 'confidence'])
    if col is None:
        raise KeyError('No confidence column found (expected pred_confidence or confidence)')
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(df[col].astype(float), bins=20, color='#4682B4')
    ax.set_title('Confidence Distribution')
    ax.set_xlabel('Confidence')
    fig.tight_layout()
    return fig


def plot_wordcloud(df: pd.DataFrame):
    text = " ".join(df['headline'].astype(str).tolist())
    if not text.strip():
        # return empty figure
        fig, ax = plt.subplots(figsize=(8,3))
        ax.text(0.5, 0.5, 'No text to generate wordcloud', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        return fig

    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8,3))
    try:
        # Some wordcloud/numpy combinations expose to_array(copy=...) which fails on some numpy versions; try safer conversion
        arr = wc.to_array()
    except TypeError:
        import numpy as _np
        arr = _np.array(wc.to_image())
    ax.imshow(arr, interpolation='bilinear')
    ax.axis('off')
    fig.tight_layout()
    return fig
