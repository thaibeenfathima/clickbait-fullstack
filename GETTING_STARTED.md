# 🚀 DeClickify - ML Integration Complete

## What Changed

✅ **Simplified Architecture**
- Removed complex API framework
- Integrated ML model directly into frontend via lightweight Flask server
- Cleaner, more efficient codebase

✅ **Direct ML Model Integration**
- Frontend connects directly to ML inference server
- No intermediate API gateway
- Faster response times

✅ **Removed Files**
- `api_server.py` - Old full-featured API
- `run_api.py` - API runner
- `app.py` - Streamlit app (deprecated)
- Various test/import files
- `startup.bat` - Old startup script

## Quick Start

### Option 1: Using Batch Files (Windows)

**Step 1: Clean up old files**
```bash
cleanup_old_files.bat
```

**Step 2: Terminal 1 - Start ML Server**
```bash
start_ml_server.bat
```

**Step 3: Terminal 2 - Start Frontend**
```bash
cd frontend
npm install
npm run dev
```

### Option 2: Using PowerShell

**Step 1: Start ML Server**
```powershell
python ml_server.py
```

**Step 2: Start Frontend (new terminal)**
```powershell
cd frontend
npm install
npm run dev
```

### Option 3: Using Terminal Commands

**Terminal 1 - ML Server:**
```bash
python ml_server.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Architecture

```
┌─────────────────┐
│    Frontend     │
│  (React/Vite)   │──────────────┐
└─────────────────┘               │
                                  │
                            HTTP API Calls
                                  │
                                  ▼
                    ┌──────────────────────┐
                    │  ML Server (Flask)   │
                    │  Port: 5000          │
                    └──────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────┐
                    │  TensorFlow Models   │
                    │  - clickbait_model   │
                    │  - clickbait_bilstm  │
                    └──────────────────────┘
```

## Features

### Single Headline Analysis
- Enter a headline
- Get instant clickbait detection
- View confidence score
- See prediction breakdown

### Batch Processing
- Upload CSV files with headlines
- Process multiple headlines at once
- Download results with predictions
- See statistics (clickbait rate, etc.)

### Model Information
- Deep learning model: TensorFlow/Keras
- Preprocessing: Text tokenization and padding
- Model type: BiLSTM neural network
- Training data: ~5000 labeled headlines

## API Endpoints

### Analyze Single Headline
```
POST /api/analyze
Content-Type: application/json

{
  "headline": "This one trick doctors hate!"
}

Response:
{
  "headline": "This one trick doctors hate!",
  "is_clickbait": true,
  "confidence": 0.92,
  "models_available": true,
  "prediction": {
    "clickbait_score": 0.92,
    "non_clickbait_score": 0.08
  }
}
```

### Batch Processing
```
POST /api/batch
Content-Type: multipart/form-data

file: <CSV file with 'headline' column>

Response:
{
  "results": [
    {
      "index": 0,
      "headline": "...",
      "is_clickbait": true,
      "confidence": 0.92
    },
    ...
  ],
  "total": 100,
  "models_available": true
}
```

### Server Status
```
GET /api/status

Response:
{
  "status": "ready",
  "models_available": true
}
```

## Environment Variables

### Frontend (.env.local)
```
VITE_ML_SERVER_URL=http://localhost:5000/api
```

## Troubleshooting

### 1. ML Server Not Running
**Error:** "Cannot connect to ML server"

**Solution:**
```bash
python ml_server.py
```
Make sure port 5000 is available.

### 2. Models Not Found
**Error:** "Models not loaded"

**Solution:**
Check that these files exist:
- `models/clickbait_model.h5`
- `models/tokenizer.pkl`
- `models/clickbait_bilstm.h5`

### 3. Frontend Won't Load
**Error:** "Cannot GET /api/analyze"

**Solution:**
1. Make sure ML server is running (`python ml_server.py`)
2. Frontend should be on port 5173
3. ML server should be on port 5000

### 4. Port Already in Use
**Error:** "Address already in use"

**Solution:**
Kill the process using the port:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :5000
kill -9 <PID>
```

## Building for Production

### Frontend
```bash
cd frontend
npm run build
```

Creates optimized build in `frontend/dist/`

### ML Server
For production deployment, use a proper WSGI server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 ml_server:app
```

## File Structure

```
DeClickify/
├── ml_server.py              # Lightweight Flask ML inference server
├── start_ml_server.bat       # Batch script to start ML server
├── start_ml_server.ps1       # PowerShell script to start ML server
├── cleanup_old_files.bat     # Remove old API-related files
├── FRONTEND_STARTUP_GUIDE.md # Frontend setup instructions
├── GETTING_STARTED.md        # This file
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Analyze.jsx         # Single headline analysis
│   │   │   ├── BatchUpload.jsx     # Batch processing
│   │   │   └── ...
│   │   ├── services/
│   │   │   └── api.js              # ML server client
│   │   └── ...
│   └── ...
│
├── models/
│   ├── clickbait_model.h5
│   ├── clickbait_bilstm.h5
│   ├── tokenizer.pkl
│   └── ...
│
├── src/
│   ├── predict.py                # Model prediction logic
│   ├── preprocess.py             # Text preprocessing
│   └── ...
│
└── data/
    └── clickbait_data.csv
```

## Next Steps

1. ✅ Run cleanup: `cleanup_old_files.bat`
2. ✅ Start ML server: `start_ml_server.bat`
3. ✅ Start frontend: `cd frontend && npm run dev`
4. ✅ Open browser: http://localhost:5173
5. ✅ Start analyzing headlines!

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all models exist in `models/` directory
3. Ensure both servers are running on correct ports
4. Check browser console for error messages

Enjoy DeClickify! 🎉
