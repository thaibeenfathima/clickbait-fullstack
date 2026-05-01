# DeClickify Frontend Setup & Startup Guide

## Quick Start

The frontend uses Vite with React. Follow these steps:

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start the Frontend Development Server
```bash
npm run dev
```

The frontend will start on `http://localhost:5173` (or similar).

### 3. Start the ML Server (in a separate terminal)
```bash
# From the root directory
python ml_server.py
```

Or use one of the startup scripts:
- **Windows Batch:** `start_ml_server.bat`
- **PowerShell:** `start_ml_server.ps1`

## Environment Variables

Create a `.env.local` file in the frontend directory:

```
VITE_ML_SERVER_URL=http://localhost:5000/api
```

## Features

- **Single Headline Analysis:** Analyze individual headlines for clickbait detection
- **Batch Processing:** Upload CSV files with multiple headlines for batch analysis
- **Direct ML Model:** Uses the TensorFlow/Keras deep learning model directly
- **Real-time Results:** Instant predictions with confidence scores

## Build for Production

```bash
npm run build
```

This creates an optimized build in the `dist/` directory.

## Troubleshooting

### ML Server Not Found
If you get a "Cannot connect to ML server" error:
1. Make sure `python ml_server.py` is running
2. Check that port 5000 is available
3. Verify the ML models exist in the `models/` directory

### Model Not Loading
If the ML server shows "Models not loaded":
1. Ensure `models/clickbait_model.h5` exists
2. Ensure `models/tokenizer.pkl` exists
3. Check that TensorFlow and Keras are installed

## Architecture

The application consists of:
- **Frontend:** React + Vite + Tailwind CSS
- **ML Server:** Lightweight Flask server with TensorFlow/Keras models
- **Models:** Pre-trained deep learning models for clickbait detection

The frontend communicates directly with the ML server (no complex API gateway).
