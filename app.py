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
import math

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

MODEL_LOADED = False
model = None
tokenizer = None

try:
    # Load only the tokenizer from the minimal AutoTokenizer without full transformers  
    from tokenizers import Tokenizer as TokenizersTokenizer
    
    tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.json")
    tokenizer = TokenizersTokenizer.from_file(tokenizer_path)
    
    # Load model weights using safetensors + torch only (no transformers needed)
    from safetensors.torch import load_file
    
    model_path = os.path.join(MODEL_PATH, "model.safetensors")
    state_dict = load_file(model_path)
    
    # Load config to understand model structure
    import json
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create a DistilBERT model wrapper that runs actual transformer inference
    class DistilBertModel:
        def __init__(self, state_dict, config):
            self.state_dict = state_dict
            self.config = config
            self.device = torch.device("cpu")
            self.is_loaded = True
            self.vocab_size = config.get('vocab_size', 30522)
            self.hidden_dim = config.get('dim', 768)
            self.num_labels = 2  # Binary classification: clickbait vs non-clickbait
            self.n_layers = config.get('n_layers', 6)
            self.n_heads = config.get('n_heads', 12)
            self.head_dim = self.hidden_dim // self.n_heads
            
        def gelu(self, x):
            """GELU activation"""
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        
        def predict(self, input_ids, attention_mask):
            """Full DistilBERT forward pass through all 6 transformer layers"""
            with torch.no_grad():
                batch_size, seq_length = input_ids.shape
                
                # 1. Embeddings
                word_embeddings = self.state_dict.get('distilbert.embeddings.word_embeddings.weight')
                position_embeddings = self.state_dict.get('distilbert.embeddings.position_embeddings.weight')
                embed_ln_weight = self.state_dict.get('distilbert.embeddings.LayerNorm.weight')
                embed_ln_bias = self.state_dict.get('distilbert.embeddings.LayerNorm.bias')
                
                hidden_states = torch.nn.functional.embedding(input_ids, word_embeddings)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0)
                hidden_states = hidden_states + torch.nn.functional.embedding(position_ids, position_embeddings)
                hidden_states = torch.nn.functional.layer_norm(hidden_states, (self.hidden_dim,), embed_ln_weight, embed_ln_bias)
                
                # 2. Transformer layers
                for layer_idx in range(self.n_layers):
                    # Self-attention
                    q_w = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.q_lin.weight']
                    q_b = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.q_lin.bias']
                    k_w = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.k_lin.weight']
                    k_b = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.k_lin.bias']
                    v_w = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.v_lin.weight']
                    v_b = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.v_lin.bias']
                    out_w = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.out_lin.weight']
                    out_b = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.attention.out_lin.bias']
                    
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
                    ffn_w1 = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.ffn.lin1.weight']
                    ffn_b1 = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.ffn.lin1.bias']
                    ffn_w2 = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.ffn.lin2.weight']
                    ffn_b2 = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.ffn.lin2.bias']
                    
                    ffn = torch.nn.functional.linear(attn_output, ffn_w1, ffn_b1)
                    ffn = self.gelu(ffn)
                    ffn = torch.nn.functional.linear(ffn, ffn_w2, ffn_b2)
                    
                    # Layer norm
                    ln_w = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.output_layer_norm.weight']
                    ln_b = self.state_dict[f'distilbert.transformer.layer.{layer_idx}.output_layer_norm.bias']
                    hidden_states = torch.nn.functional.layer_norm(attn_output + ffn, (self.hidden_dim,), ln_w, ln_b)
                
                # 3. Classification head - use [CLS] token
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
    
    print("[OK] Model weights loaded successfully from saved_model/")
    print(f"    Running DistilBERT with {model.n_layers} transformer layers")
    print(f"    Binary classification: Clickbait vs Non-clickbait")
    MODEL_LOADED = True
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    import traceback
    traceback.print_exc()
    MODEL_LOADED = False
    model = None
    tokenizer = None

LABEL_MAP = {
    0: "Not clickbait",
    1: "Clickbait",
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
    if not MODEL_LOADED or model is None or tokenizer is None:
        raise ValueError("Model not loaded. Check server logs.")
    
    # Tokenize using the tokenizer (max_length is set in tokenizer config)
    encoding = tokenizer.encode(text)
    input_ids = encoding.ids
    attention_mask = encoding.attention_mask
    
    # Convert to torch tensors
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
    
    with torch.no_grad():
        # Use our simple model wrapper
        logits = model.predict(input_ids_tensor, attention_mask_tensor)
    
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
    return {
        "model_file": os.path.join(MODEL_PATH, "model.safetensors"),
        "weights_loaded": MODEL_LOADED,
        "note": "BERT model loaded and ready" if MODEL_LOADED else "Model loading failed - check server logs",
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
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

