# DeClickify Architecture & System Overview

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        WEB BROWSER                            │
│                   (User Interface)                            │
└────────────────────────────┬─────────────────────────────────┘
                             │
                    HTTP/CORS │ Port 5173
                             │
         ┌───────────────────▼────────────────────┐
         │      Frontend Application              │
         │   (React + Vite + Tailwind CSS)        │
         │                                         │
         │  ┌──────────────────────────────────┐  │
         │  │  Pages                           │  │
         │  │  - Analyze.jsx (single)          │  │
         │  │  - BatchUpload.jsx (CSV)         │  │
         │  │  - Home.jsx                      │  │
         │  │  - About.jsx                     │  │
         │  └──────────────────────────────────┘  │
         │                                         │
         │  ┌──────────────────────────────────┐  │
         │  │  Services                        │  │
         │  │  - api.js (ML Server Client)     │  │
         │  └──────────────────────────────────┘  │
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                    REST API  │ Port 5000
                    JSON      │
                              │
         ┌────────────────────▼────────────────────┐
         │     ML Inference Server                 │
         │      (Flask + Python)                   │
         │     ml_server.py                        │
         │                                         │
         │  ┌──────────────────────────────────┐  │
         │  │  Endpoints                       │  │
         │  │  POST /api/analyze               │  │
         │  │  POST /api/batch                 │  │
         │  │  GET /api/status                 │  │
         │  └──────────────────────────────────┘  │
         │                                         │
         └────────────────────┬────────────────────┘
                              │
              Python Inference│
              TensorFlow/Keras│
                              │
         ┌────────────────────▼────────────────────┐
         │    Machine Learning Models              │
         │                                         │
         │  ┌──────────────────────────────────┐  │
         │  │ Keras Model                      │  │
         │  │ clickbait_model.h5               │  │
         │  │                                  │  │
         │  │ Input: Text (tokenized)          │  │
         │  │ Output: Clickbait probability    │  │
         │  └──────────────────────────────────┘  │
         │                                         │
         │  ┌──────────────────────────────────┐  │
         │  │ Tokenizer                        │  │
         │  │ tokenizer.pkl                    │  │
         │  │                                  │  │
         │  │ Converts text → token sequences  │  │
         │  └──────────────────────────────────┘  │
         │                                         │
         └─────────────────────────────────────────┘
```

## Data Flow: Single Headline Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER ENTERS TEXT                            │
│                 "This one trick doctors hate!"                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND: Analyze.jsx Component                     │
│                   - Capture user input                          │
│                   - Submit to ML server                         │
│                   - Show loading state                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                   POST /api/analyze
                 { headline: "..." }
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              ML SERVER: Receive Request                          │
│                   - Parse JSON body                             │
│                   - Extract headline text                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              TEXT PREPROCESSING                                  │
│  - Lowercase conversion                                          │
│  - Remove special characters                                    │
│  - Remove extra whitespace                                      │
│  - Output: "this one trick doctors hate"                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              TOKENIZATION (Keras Tokenizer)                     │
│  - Convert words to token IDs                                   │
│  - Max vocabulary: NUM_WORDS                                    │
│  - Output: [423, 1, 567, 234, 123]                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              PADDING/TRUNCATION                                  │
│  - Pad sequences to MAX_LEN (e.g., 100 tokens)                 │
│  - Output: [423, 1, 567, 234, 123, 0, 0, ..., 0]              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL PREDICTION                                    │
│  - Input: Padded token sequence                                 │
│  - BiLSTM layers process sequence                              │
│  - Dense layer + Sigmoid activation                            │
│  - Output: [0.92] (92% probability of clickbait)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FORMAT RESPONSE (JSON)                              │
│  {                                                               │
│    "headline": "This one trick doctors hate!",                 │
│    "is_clickbait": true,                                        │
│    "confidence": 0.92,                                          │
│    "models_available": true,                                    │
│    "prediction": {                                              │
│      "clickbait_score": 0.92,                                   │
│      "non_clickbait_score": 0.08                                │
│    }                                                             │
│  }                                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
               Return JSON response
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND: Display Results                           │
│  - Parse response JSON                                          │
│  - Show classification badge                                    │
│  - Display confidence progress bar                              │
│  - Render visual indicators                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 USER SEES RESULTS                                │
│         ⚠️ Clickbait Detected - 92% Confidence                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow: Batch Processing

```
┌──────────────────────────────────────────────────────────────┐
│                   USER UPLOADS CSV FILE                      │
│    Headline                                                   │
│    This one trick will change your life                       │
│    10 reasons why cats are better than dogs                  │
│    Scientists discover new species                           │
│    ...                                                        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│         FRONTEND: BatchUpload.jsx Component                  │
│    - Read file                                               │
│    - Show preview (first 5 rows)                            │
│    - User selects headline column                            │
│    - Click "Process Batch"                                   │
└──────────────────────────┬───────────────────────────────────┘
                           │
              POST /api/batch (multipart)
                  file: <CSV data>
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│        ML SERVER: Batch Endpoint                             │
│    - Parse CSV file                                          │
│    - Iterate through rows                                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │                                      │
        │  For each headline (row):           │
        │  1. Preprocess                      │
        │  2. Tokenize                        │
        │  3. Pad/Truncate                    │
        │  4. Predict                         │
        │  5. Append to results               │
        │                                      │
        └──────────────────┬──────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│        FORMAT BATCH RESPONSE (JSON)                          │
│  {                                                            │
│    "results": [                                              │
│      {                                                        │
│        "index": 0,                                            │
│        "headline": "...",                                    │
│        "is_clickbait": true,                                │
│        "confidence": 0.92                                    │
│      },                                                       │
│      ...                                                      │
│    ],                                                         │
│    "total": 100,                                             │
│    "models_available": true                                  │
│  }                                                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
               Return batch results JSON
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│         FRONTEND: Display Results                            │
│    - Show statistics (total, clickbait %, etc.)             │
│    - Display results table (first 10 rows)                  │
│    - Provide CSV download button                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              USER DOWNLOADS RESULTS CSV                      │
│    Headline,Classification,Confidence                        │
│    "This one trick...",⚠️ Clickbait,92%                     │
│    "Scientists discover...",✓ Legitimate,15%                │
│    ...                                                        │
└──────────────────────────────────────────────────────────────┘
```

## File Organization

```
DeClickify/
│
├── 📄 ml_server.py                          ⭐ MAIN - ML inference server
├── 📄 start_ml_server.bat                   Quick start (Windows)
├── 📄 start_ml_server.ps1                   Quick start (PowerShell)
├── 📄 cleanup_old_files.bat                 Remove deprecated files
│
├── 📂 frontend/                             React + Vite application
│   ├── 📄 package.json
│   ├── 📄 vite.config.js
│   ├── 📄 tailwind.config.js
│   │
│   └── 📂 src/
│       ├── 📂 pages/
│       │   ├── Analyze.jsx                 ⭐ Single headline analysis
│       │   ├── BatchUpload.jsx              ⭐ Batch CSV processing
│       │   ├── Home.jsx
│       │   ├── About.jsx
│       │   └── Dashboard.jsx
│       │
│       ├── 📂 components/
│       │   ├── Card.jsx
│       │   ├── Button.jsx
│       │   ├── Badge.jsx
│       │   ├── Navbar.jsx
│       │   └── Footer.jsx
│       │
│       └── 📂 services/
│           └── api.js                      ⭐ ML server client
│
├── 📂 models/                               ⭐ TensorFlow Models
│   ├── clickbait_model.h5                   Main classification model
│   ├── clickbait_bilstm.h5                  Alternative model
│   ├── tokenizer.pkl                        Text tokenizer
│   ├── X_train.npy                          Training data
│   ├── X_test.npy                           Test data
│   ├── y_train.npy                          Training labels
│   └── y_test.npy                           Test labels
│
├── 📂 src/                                  Core ML utilities
│   ├── predict.py                           Prediction logic
│   ├── preprocess.py                        Text preprocessing
│   ├── train_model.py                       Model training
│   ├── batch_processor.py                   Batch processing utilities
│   └── ...
│
├── 📂 data/
│   └── clickbait_data.csv                   Training dataset
│
└── 📂 scripts/                              Utility scripts
```

## Request/Response Examples

### Single Headline Analysis
**Request:**
```http
POST http://localhost:5000/api/analyze
Content-Type: application/json

{
  "headline": "This one trick will change your life!"
}
```

**Response:**
```json
{
  "headline": "This one trick will change your life!",
  "is_clickbait": true,
  "confidence": 0.95,
  "models_available": true,
  "prediction": {
    "clickbait_score": 0.95,
    "non_clickbait_score": 0.05
  }
}
```

### Batch Processing
**Request:**
```http
POST http://localhost:5000/api/batch
Content-Type: multipart/form-data

file: <CSV file>
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "headline": "Scientists discover new species",
      "is_clickbait": false,
      "confidence": 0.15
    },
    {
      "index": 1,
      "headline": "This photo will make you cry",
      "is_clickbait": true,
      "confidence": 0.88
    }
  ],
  "total": 2,
  "models_available": true
}
```

### Server Status
**Request:**
```http
GET http://localhost:5000/api/status
```

**Response:**
```json
{
  "status": "ready",
  "models_available": true
}
```

## Component Communication

```
┌──────────────────────────────────────────────────────┐
│  Browser (Render Process)                            │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │ Analyze.jsx / BatchUpload.jsx                  │ │
│  │                                                 │ │
│  │  ┌──────────────────────────────────────────┐ │ │
│  │  │ Component State (React Hooks)            │ │ │
│  │  │ - headline (text input)                  │ │ │
│  │  │ - result (API response)                  │ │ │
│  │  │ - loading (true/false)                   │ │ │
│  │  │ - error (error message)                  │ │ │
│  │  └──────────────────────────────────────────┘ │ │
│  │                   │                            │ │
│  │                   ▼                            │ │
│  │  ┌──────────────────────────────────────────┐ │ │
│  │  │ api.js (Service Layer)                   │ │ │
│  │  │ - analyzeHeadline()                      │ │ │
│  │  │ - processBatch()                         │ │ │
│  │  │ - checkServerStatus()                    │ │ │
│  │  └──────────────────────────────────────────┘ │ │
│  │                   │                            │ │
│  │                   ▼                            │ │
│  │  ┌──────────────────────────────────────────┐ │ │
│  │  │ Axios HTTP Client                        │ │ │
│  │  │ - baseURL: http://localhost:5000/api     │ │ │
│  │  │ - headers: application/json              │ │ │
│  │  └──────────────────────────────────────────┘ │ │
│  │                   │                            │ │
│  │                   ▼                            │ │
│  │           Network Request (HTTP)              │ │
│  │                                                 │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
└──────────────────────────────────────────────────────┘
              ║                          ║
              ║ HTTP/JSON                ║
         POST /api/analyze           Response JSON
              ║                          ║
              ▼                          ▲
┌──────────────────────────────────────────────────────┐
│  Server Process (ml_server.py)                       │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │ Flask Application                              │ │
│  │ - Route: /api/analyze                          │ │
│  │ - Route: /api/batch                            │ │
│  │ - Route: /api/status                           │ │
│  └────────────────────────────────────────────────┘ │
│                   │                                 │
│                   ▼                                 │
│  ┌────────────────────────────────────────────────┐ │
│  │ Data Processing                                │ │
│  │ - Text preprocessing                           │ │
│  │ - Tokenization                                 │ │
│  │ - Padding/Truncation                           │ │
│  └────────────────────────────────────────────────┘ │
│                   │                                 │
│                   ▼                                 │
│  ┌────────────────────────────────────────────────┐ │
│  │ Model Inference (TensorFlow/Keras)             │ │
│  │ - Load model: clickbait_model.h5               │ │
│  │ - Run prediction                               │ │
│  │ - Get probability score                        │ │
│  └────────────────────────────────────────────────┘ │
│                   │                                 │
│                   ▼                                 │
│  ┌────────────────────────────────────────────────┐ │
│  │ Response Formatting (JSON)                     │ │
│  │ - is_clickbait (boolean)                       │ │
│  │ - confidence (0.0-1.0)                         │ │
│  │ - models_available (boolean)                   │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Performance Timeline

```
User Action                     Time            Status
─────────────────────────────────────────────────────────
1. Enter headline              0ms              Ready
2. Click "Analyze"             0ms              Loading
3. HTTP Request sent           5ms              In-flight
4. Server receives request     10ms             Processing
5. Text preprocessing          15ms             Processing
6. Tokenization                20ms             Processing
7. Padding                     25ms             Processing
8. Model inference             75ms             Processing
9. Response formatting         85ms             Processing
10. HTTP Response sent         90ms             Complete
11. Frontend receives response 95ms             Rendering
12. UI updates                 110ms            ✓ Complete

Total Time: ~110ms (depends on network & hardware)
```

This diagram illustrates the complete flow and architecture of the DeClickify system!
