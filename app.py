# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import json
import os
import re
import math
import traceback

# Suppress TensorFlow logs if TF is installed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# -----------------------
# Project paths - robust search for saved_model
# -----------------------
BASE_DIR = os.path.dirname(__file__)

candidate_paths = [
    os.path.join(BASE_DIR, "saved_model"),
    os.path.join(BASE_DIR, "backend", "saved_model"),
    os.path.join(BASE_DIR, "server", "saved_model"),
    os.path.join(BASE_DIR, "api", "saved_model"),
    os.path.join(BASE_DIR, "..", "saved_model"),  # parent folder
]

MODEL_PATH = None
for p in candidate_paths:
    if os.path.isdir(p):
        MODEL_PATH = os.path.abspath(p)
        break

if MODEL_PATH is None:
    MODEL_PATH = os.path.join(BASE_DIR, "saved_model")

print(f"[INFO] Using MODEL_PATH = {MODEL_PATH}")
print("[INFO] Contents of MODEL_PATH (if exists):")
if os.path.isdir(MODEL_PATH):
    for fn in sorted(os.listdir(MODEL_PATH)):
        print(" -", fn)
else:
    print("  (none)")

FRONTEND_DIST = os.path.join(BASE_DIR, "dist")  # React build will be copied here

app = FastAPI(title="Clickbait Detection API + Frontend")

# CORS – in production restrict this to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting Clickbait Detection API...")

# ===============================
# LOAD MODEL AND TOKENIZER (robust)
# ===============================
print("Loading model and tokenizer...")

MODEL_LOADED = False
model = None
tokenizer = None
_tokenizer_type = None  # "transformers" or "tokenizers" or None

try:
    # First try the transformers route (most robust if model dir saved with HF)
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("Attempting to load with transformers.AutoModelForSequenceClassification.from_pretrained...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        model.to(torch.device("cpu"))
        model.eval()
        MODEL_LOADED = True
        _tokenizer_type = "transformers"
        print("[OK] Loaded model and tokenizer via transformers from:", MODEL_PATH)
    except Exception as e_tr:
        print("[INFO] transformers load failed (will attempt safetensors/tokenizers fallback):", repr(e_tr))
        # Fallback: tokenizers + safetensors custom loader
        from tokenizers import Tokenizer as TokenizersTokenizer
        from safetensors.torch import load_file as safetensors_load

        # load tokenizer.json (tokenizers library)
        tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = TokenizersTokenizer.from_file(tokenizer_path)
            _tokenizer_type = "tokenizers"
            print("[OK] Tokenizer loaded from tokenizer.json")
        else:
            raise FileNotFoundError(f"tokenizer.json not found in {MODEL_PATH}")

        # locate a safetensors or pytorch weights file
        model_file_safetensors = os.path.join(MODEL_PATH, "model.safetensors")
        model_file_bin = os.path.join(MODEL_PATH, "pytorch_model.bin")
        model_file_pt = os.path.join(MODEL_PATH, "pytorch_model.pt")
        found_weights = None
        if os.path.exists(model_file_safetensors):
            found_weights = model_file_safetensors
            state_dict = safetensors_load(found_weights)
            print(f"[OK] safetensors state_dict loaded, keys count: {len(state_dict)}")
        elif os.path.exists(model_file_bin) or os.path.exists(model_file_pt):
            # try torch.load on .bin/.pt
            found_weights = model_file_bin if os.path.exists(model_file_bin) else model_file_pt
            print(f"[INFO] Found weights file {found_weights}, attempting torch.load...")
            state_dict = torch.load(found_weights, map_location="cpu")
            # if it's a dict with 'state_dict' key, extract
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            print(f"[OK] torch.load returned keys count: {len(state_dict)}")
        else:
            raise FileNotFoundError(f"No weights file found in {MODEL_PATH}. Expected model.safetensors or pytorch_model.bin/.pt")

        # Ensure torch tensors on CPU
        for k, v in list(state_dict.items()):
            if not isinstance(v, torch.Tensor):
                try:
                    state_dict[k] = torch.tensor(v)
                except Exception:
                    # leave as-is
                    pass
            else:
                state_dict[k] = v.to(torch.device("cpu"))

        # load config.json if present
        config_path = os.path.join(MODEL_PATH, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}
            print("[WARN] No config.json found in model dir; using defaults")

        # Minimal DistilBERT wrapper with guards
        class DistilBertModel:
            def __init__(self, state_dict, config):
                self.state_dict = state_dict
                self.config = config
                self.device = torch.device("cpu")
                self.is_loaded = True
                self.vocab_size = config.get('vocab_size', 30522)
                self.hidden_dim = config.get('dim', 768)
                self.num_labels = config.get('num_labels', 2) or 2
                self.n_layers = config.get('n_layers', 6)
                self.n_heads = config.get('n_heads', 12)
                self.head_dim = self.hidden_dim // self.n_heads

            def gelu(self, x):
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            def has(self, key):
                return key in self.state_dict and self.state_dict.get(key) is not None

            def predict(self, input_ids, attention_mask):
                required_keys = [
                    'distilbert.embeddings.word_embeddings.weight',
                    'distilbert.embeddings.position_embeddings.weight',
                    'distilbert.embeddings.LayerNorm.weight',
                    'distilbert.embeddings.LayerNorm.bias',
                    'pre_classifier.weight',
                    'pre_classifier.bias',
                    'classifier.weight',
                    'classifier.bias'
                ]
                missing = [k for k in required_keys if not self.has(k)]
                if missing:
                    raise RuntimeError(f"Missing required model keys in state_dict: {missing}")

                with torch.no_grad():
                    batch_size, seq_length = input_ids.shape
                    word_embeddings = self.state_dict['distilbert.embeddings.word_embeddings.weight']
                    position_embeddings = self.state_dict['distilbert.embeddings.position_embeddings.weight']
                    embed_ln_weight = self.state_dict['distilbert.embeddings.LayerNorm.weight']
                    embed_ln_bias = self.state_dict['distilbert.embeddings.LayerNorm.bias']

                    hidden_states = torch.nn.functional.embedding(input_ids, word_embeddings)
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0)
                    hidden_states = hidden_states + torch.nn.functional.embedding(position_ids, position_embeddings)
                    hidden_states = torch.nn.functional.layer_norm(hidden_states, (self.hidden_dim,), embed_ln_weight, embed_ln_bias)

                    for layer_idx in range(self.n_layers):
                        base = f'distilbert.transformer.layer.{layer_idx}'
                        layer_keys = [
                            f'{base}.attention.q_lin.weight',
                            f'{base}.attention.k_lin.weight',
                            f'{base}.attention.v_lin.weight',
                            f'{base}.attention.out_lin.weight',
                            f'{base}.ffn.lin1.weight',
                            f'{base}.ffn.lin2.weight',
                            f'{base}.output_layer_norm.weight'
                        ]
                        if any(not self.has(k) for k in layer_keys):
                            raise RuntimeError(f"Missing layer parameters for transformer layer {layer_idx}. Keys checked: {layer_keys}")

                        q_w = self.state_dict[f'{base}.attention.q_lin.weight']
                        q_b = self.state_dict.get(f'{base}.attention.q_lin.bias', None)
                        k_w = self.state_dict[f'{base}.attention.k_lin.weight']
                        k_b = self.state_dict.get(f'{base}.attention.k_lin.bias', None)
                        v_w = self.state_dict[f'{base}.attention.v_lin.weight']
                        v_b = self.state_dict.get(f'{base}.attention.v_lin.bias', None)
                        out_w = self.state_dict[f'{base}.attention.out_lin.weight']
                        out_b = self.state_dict.get(f'{base}.attention.out_lin.bias', None)

                        Q = torch.nn.functional.linear(hidden_states, q_w, q_b)
                        K = torch.nn.functional.linear(hidden_states, k_w, k_b)
                        V = torch.nn.functional.linear(hidden_states, v_w, v_b)

                        Q = Q.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
                        K = K.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
                        V = V.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

                        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        if attention_mask is not None:
                            mask = (attention_mask.unsqueeze(1).unsqueeze(2) == 1)
                            scores = scores.masked_fill(~mask, float('-inf'))

                        attn_weights = torch.softmax(scores, dim=-1)
                        context = torch.matmul(attn_weights, V)
                        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
                        attn_output = torch.nn.functional.linear(context, out_w, out_b)

                        # Feed-forward
                        ffn_w1 = self.state_dict[f'{base}.ffn.lin1.weight']
                        ffn_b1 = self.state_dict.get(f'{base}.ffn.lin1.bias', None)
                        ffn_w2 = self.state_dict[f'{base}.ffn.lin2.weight']
                        ffn_b2 = self.state_dict.get(f'{base}.ffn.lin2.bias', None)

                        ffn = torch.nn.functional.linear(attn_output, ffn_w1, ffn_b1)
                        ffn = self.gelu(ffn)
                        ffn = torch.nn.functional.linear(ffn, ffn_w2, ffn_b2)

                        ln_w = self.state_dict[f'{base}.output_layer_norm.weight']
                        ln_b = self.state_dict[f'{base}.output_layer_norm.bias']
                        hidden_states = torch.nn.functional.layer_norm(attn_output + ffn, (self.hidden_dim,), ln_w, ln_b)

                    cls_output = hidden_states[:, 0, :]
                    pre_clf_w = self.state_dict['pre_classifier.weight']
                    pre_clf_b = self.state_dict['pre_classifier.bias']
                    clf_feat = torch.nn.functional.linear(cls_output, pre_clf_w, pre_clf_b)
                    clf_feat = torch.relu(clf_feat)

                    clf_w = self.state_dict['classifier.weight']
                    clf_b = self.state_dict['classifier.bias']
                    logits = torch.nn.functional.linear(clf_feat, clf_w, clf_b)

                    return logits

        model = DistilBertModel(state_dict, config)
        MODEL_LOADED = True
        print("[OK] Loaded safetensors/tokenizers fallback model from saved_model/")
except Exception as e:
    print(f"[ERROR] Could not load model/tokenizer: {e}")
    traceback.print_exc()
    MODEL_LOADED = False
    model = None
    tokenizer = None
    _tokenizer_type = None

# ===============================
# END model loading
# ===============================

# ----------------------------
# Load a real sentiment model (Hugging Face) for better sentiment results
# ----------------------------
_sentiment_pipe = None
try:
    from transformers import pipeline
    print("[INFO] Loading sentiment pipeline (distilbert-base-uncased-finetuned-sst-2-english)...")
    _sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU inference; set to 0 if you want GPU and have CUDA
    )
    print("[OK] Sentiment pipeline loaded.")
except Exception as e:
    print("[WARN] Could not load sentiment pipeline:", e)
    _sentiment_pipe = None

LABEL_MAP = {
    0: "Not clickbait",
    1: "Clickbait",
}

# ===============================
# SENTIMENT ANALYSIS (pipeline-backed)
# ===============================
def get_sentiment(text: str):
    """
    Use the HF sentiment pipeline if available. Returns:
    { "category": "positive|negative|neutral", "label": "Positive|Negative|Neutral", "score": float }
    - neutral is returned only when pipeline fails or model low confidence (~0.5)
    """
    text = (text or "").strip()
    if text == "":
        return {"category": "neutral", "label": "Neutral", "score": 0.5}

    # Prefer the real model if available
    global _sentiment_pipe
    try:
        if _sentiment_pipe is not None:
            out = _sentiment_pipe(text[:512])  # truncate for safety
            # out is a list like [{'label': 'POSITIVE', 'score': 0.9998}]
            if isinstance(out, list) and len(out) > 0 and "label" in out[0]:
                lab = out[0]["label"]
                score = float(out[0].get("score", 0.5))
                if lab.upper().startswith("POS"):
                    return {"category": "positive", "label": "Positive", "score": score}
                elif lab.upper().startswith("NEG"):
                    return {"category": "negative", "label": "Negative", "score": score}
                else:
                    return {"category": "neutral", "label": lab.title(), "score": score}
    except Exception as e:
        # Log but fall back to simple neutral
        print("[WARN] sentiment pipeline error:", e)

    # Fallback: neutral if sentiment pipeline unavailable or failed
    return {"category": "neutral", "label": "Neutral", "score": 0.5}

# ===============================
# PREDICTION LOGIC
# ===============================
def predict_text(text: str):
    if not MODEL_LOADED or model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Check server logs and ensure model and tokenizer exist in saved_model/")

    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("Empty headline provided")

    # Tokenize according to tokenizer type
    if _tokenizer_type == "transformers":
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        input_ids_tensor = enc["input_ids"].to(torch.device("cpu"))
        attention_mask_tensor = enc["attention_mask"].to(torch.device("cpu"))
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    elif _tokenizer_type == "tokenizers":
        encoding = tokenizer.encode(text)
        input_ids = encoding.ids
        attention_mask = getattr(encoding, "attention_mask", [1] * len(input_ids))
        # pad/truncate to model max length
        max_len = 128
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
        else:
            pad_len = max_len - len(input_ids)
            pad_id = 0
            try:
                pad_id = tokenizer.token_to_id("[PAD]")
            except Exception:
                # fallback
                pad_id = 0
            input_ids = input_ids + ([pad_id] * pad_len)
            attention_mask = attention_mask + ([0] * pad_len)

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
        with torch.no_grad():
            logits = model.predict(input_ids_tensor, attention_mask_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    else:
        raise RuntimeError("Unknown tokenizer/model type. Check server model loader.")

    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    LABEL_STR = {
        0: "not_clickbait",
        1: "clickbait"
    }

    prediction = LABEL_MAP.get(pred_id, "Unknown")
    label = LABEL_STR.get(pred_id, str(pred_id))

    probs_dict = {
        "not_clickbait": float(probs[0]) if len(probs) > 0 else 0.0,
        "clickbait": float(probs[1]) if len(probs) > 1 else 0.0
    }

    sentiment = get_sentiment(text)
    highlighted_words = [w for w in re.findall(r"\w+", text) if len(w) > 4][:3]

    return {
        "prediction": prediction,        # legacy label
        "label_id": pred_id,             # numeric id (0/1)
        "label": label,                  # standardized label
        "probs": probs_dict,             # both class probabilities
        "confidence": confidence,        # probability for predicted class
        "sentiment": sentiment,          # now a dict: {category,label,score}
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
        result = predict_text(str(text))
        result["headline"] = text
        results.append(result)
    return results

@app.get("/api/model_status")
def model_status():
    return {
        "model_path": MODEL_PATH,
        "weights_present": any(os.path.exists(os.path.join(MODEL_PATH, fn)) for fn in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt")),
        "weights_loaded": MODEL_LOADED,
        "tokenizer_type": _tokenizer_type,
        "note": "Model loaded" if MODEL_LOADED else "Model loading failed - check server logs",
    }

# ========== FRONTEND STATIC (React dist) ==========
if os.path.isdir(FRONTEND_DIST):
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="frontend",
    )

# ========== RUN (if executed directly) ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
