# DeClickify

DeClickify is a Streamlit app to detect clickbait headlines and provide sentiment and explainability.

What's new:
- File-hash caching for uploaded files: parsing results are cached by content hash to avoid re-parsing identical uploads.
- Integrated Gradients (IG) explainability implemented in `src/explainability.py` with Streamlit caching (`integrated_gradients(text, steps=20)`).

Notes and next steps:
- SHAP support is planned as an optional extension — IG is implemented as a lightweight, dependency-free method for token-level attribution.
- You can run quick smoke tests in `scripts/test_batch_cache.py` and `scripts/test_ig.py`.
