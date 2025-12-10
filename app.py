from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import json
import os
import re

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")  # React build will be copied here

app = FastAPI(title="Clickbait Detection API + Frontend")

# CORS – in production you can restrict this to your exact domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: ["https://your-domain.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# LOAD MODEL AND TOKENIZER
# ===============================
print("Loading model and tokenizer...")

# Load config
with open(os.path.join(MODEL_PATH, "config.json"), encoding="utf-8") as f:
    model_config = json.load(f)

# Load vocab
with open(os.path.join(MODEL_PATH, "vocab.txt"), encoding="utf-8") as f:
    vocab = [line.strip() for line in f]


class SimpleTokenizer:
    def __init__(self, vocab_list):
        self.vocab = {word: idx for idx, word in enumerate(vocab_list)}
        # Special tokens
        self.vocab['[CLS]'] = 101
        self.vocab['[SEP]'] = 102
        self.vocab['[PAD]'] = 0
        self.vocab['[UNK]'] = 100
        self.max_length = 512

    def tokenize(self, text):
        """Simple whitespace tokenization"""
        words = text.lower().split()
        tokens = ['[CLS]']
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in self.vocab:
                tokens.append(clean_word)
            else:
                tokens.append('[UNK]')
        tokens.append('[SEP]')
        # truncate
        return tokens[:self.max_length]

    def encode(self, text):
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        padding = [self.vocab['[PAD]']] * (self.max_length - len(ids))
        return ids + padding

    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
        ids = self.encode(text)
        if return_tensors == "pt":
            input_ids = torch.tensor([ids])
            attention_mask = torch.tensor([[1 if i != self.vocab['[PAD]'] else 0 for i in ids]])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": [ids]}


tokenizer = SimpleTokenizer(vocab)

# Load model weights from safetensors
try:
    from safetensors.torch import load_file
    model_weights = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
    print("✅ Model loaded from safetensors")
except ImportError:
    print("⚠️ safetensors not available, using fallback mode")
    model_weights = None


class SimpleSequenceClassifier(torch.nn.Module):
    def __init__(self, vocab_size=30522, num_labels=4):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        embedded = self.embeddings(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(embedded.size()).float()
            pooled = (embedded * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = embedded.mean(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return type("Output", (), {"logits": logits})()


model = SimpleSequenceClassifier()

if model_weights is not None:
    try:
        model.load_state_dict(model_weights)
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load weights: {e}")

model.eval()

LABEL_MAP = {
    0: "Listicle",
    1: "Sensational",
    2: "Question",
    3: "Teaser",
}

# ===============================
# SENTIMENT ANALYSIS
# ===============================


def get_sentiment(text: str):
    positive_words = {
        "good",
        "great",
        "love",
        "best",
        "amazing",
        "awesome",
        "excellent",
        "wonderful",
        "fantastic",
        "perfect",
        "incredible",
        "stunning",
    }
    negative_words = {
        "bad",
        "hate",
        "worst",
        "terrible",
        "awful",
        "horrible",
        "poor",
        "disgusting",
        "pathetic",
        "useless",
        "annoying",
        "disappointing",
    }

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        return {"label": "POSITIVE", "score": min(0.95, 0.5 + pos_count * 0.1)}
    elif neg_count > pos_count:
        return {"label": "NEGATIVE", "score": min(0.95, 0.5 + neg_count * 0.1)}
    else:
        return {"label": "NEUTRAL", "score": 0.5}


# ===============================
# PREDICTION LOGIC
# ===============================
def predict_text(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    prediction = LABEL_MAP.get(pred_id, "Unknown")
    sentiment = get_sentiment(text)
    highlighted_words = [w for w in text.split() if len(w) > 4][:3]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "sentiment": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "highlighted_words": highlighted_words,
    }


class Headline(BaseModel):
    headline: str


# ========== API ROUTES (prefix /api) ==========

@app.get("/api")
def api_root():
    return {"status": "Backend running with ML model", "version": "2.0"}


@app.post("/api/predict")
def predict_single(data: Headline):
    return predict_text(data.headline)


@app.post("/api/predict_csv")
def predict_batch(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    results = []
    for text in df["headline"][:100]:
        result = predict_text(text)
        result["headline"] = text
        results.append(result)
    return results


@app.get("/api/model_status")
def model_status():
    weights_present = model_weights is not None
    return {
        "model_file": os.path.join(MODEL_PATH, "model.safetensors"),
        "weights_loaded": bool(weights_present),
        "note": "Model weights loaded and ready"
        if weights_present
        else "Model weights not loaded; running in fallback mode",
    }


# ========== FRONTEND STATIC (React dist) ==========

# If dist exists, serve it at root
if os.path.isdir(FRONTEND_DIST):
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="frontend",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
