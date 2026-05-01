# 🚀 DeClickify - Clickbait Detector

**Detect and analyze clickbait headlines using deep learning**

A modern web application that uses trained neural networks to identify clickbait content in news headlines with high accuracy.

## ⚡ Quick Start (2 minutes)

### 1. Start ML Server
```bash
python ml_server.py
```

### 2. Start Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```

### 3. Open Browser
Navigate to `http://localhost:5173`

## 🎯 Features

✨ **Single Headline Analysis**
- Analyze individual headlines instantly
- Get clickbait probability with confidence score
- View detailed prediction breakdown

📊 **Batch Processing**
- Upload CSV files with multiple headlines
- Process up to hundreds of headlines at once
- Download results with detailed predictions
- View statistics and trends

🧠 **Deep Learning Model**
- BiLSTM neural network trained on 5,000+ headlines
- 90%+ accuracy on test data
- Real-time inference
- Confidence scoring

🎨 **Modern UI**
- Clean, intuitive interface
- Real-time feedback
- Progress indicators
- Responsive design

## 📋 System Requirements

- Python 3.8+
- Node.js 16+
- 2GB RAM minimum
- 500MB disk space

## 🔧 Installation

### Backend Requirements
```bash
pip install flask flask-cors tensorflow keras pandas numpy
```

### Frontend Setup
```bash
cd frontend
npm install
```

## 🚀 Running the Application

### Terminal 1 - Start ML Server
```bash
python ml_server.py
```
Server will start on `http://localhost:5000`

### Terminal 2 - Start Frontend
```bash
cd frontend
npm run dev
```
Frontend will start on `http://localhost:5173`

### Access Application
Open browser to: `http://localhost:5173`

## 📖 Usage Examples

### Example 1: Single Headline Analysis
1. Go to **Analyze** page
2. Enter headline: "This one trick will change your life!"
3. Click **Analyze Headline**
4. See result: ⚠️ Clickbait Detected - 95% Confidence

### Example 2: Batch Processing
1. Go to **Batch Upload** page
2. Upload CSV file with headlines
3. Select headline column
4. Click **Process Batch**
5. View results and download CSV

## 🏗️ Architecture

```
Frontend (React/Vite)
        ↓
        ↓ HTTP JSON
        ↓
ML Server (Flask)
        ↓
        ↓ TensorFlow/Keras
        ↓
Deep Learning Model
```

## 📚 Documentation

- [Getting Started Guide](GETTING_STARTED.md) - Complete setup instructions
- [Architecture Diagram](ARCHITECTURE_DIAGRAM.md) - System design & data flow
- [Frontend Setup Guide](FRONTEND_STARTUP_GUIDE.md) - Frontend-specific instructions
- [Integration Summary](ML_INTEGRATION_SUMMARY.md) - What changed & why

## 🤖 Model Information

- **Type:** BiLSTM Neural Network
- **Framework:** TensorFlow/Keras
- **Training Data:** 5,000+ labeled headlines
- **Accuracy:** 90%+ on test set
- **Input:** Text headlines
- **Output:** Clickbait probability (0.0-1.0)

## 📊 Supported Input Formats

### Single Analysis
- Plain text headlines
- Supports special characters and emojis
- No length restrictions

### Batch Processing
- **CSV files** (.csv) ⭐ Recommended
- Excel files (.xlsx)
- JSON files (.json)
- XML files (.xml)
- Text files (.txt)

## 🔌 API Endpoints

### Single Headline
```
POST /api/analyze
{
  "headline": "Your headline here"
}
```

### Batch Processing
```
POST /api/batch
(multipart/form-data)
file: <CSV file>
```

### Server Status
```
GET /api/status
```

## 📁 Project Structure

```
DeClickify/
├── ml_server.py                 # ML inference server
├── start_ml_server.bat          # Windows startup script
├── start_ml_server.ps1          # PowerShell startup script
├── GETTING_STARTED.md           # Quick start guide
├── ARCHITECTURE_DIAGRAM.md      # System design
├── FRONTEND_STARTUP_GUIDE.md    # Frontend guide
├── ML_INTEGRATION_SUMMARY.md    # Technical summary
│
├── frontend/                    # React application
│   ├── src/
│   │   ├── pages/              # Page components
│   │   ├── components/         # Reusable components
│   │   ├── services/           # API client
│   │   └── assets/             # Images & icons
│   └── package.json
│
├── models/                      # ML models
│   ├── clickbait_model.h5      # Main model
│   ├── tokenizer.pkl            # Text tokenizer
│   └── ...
│
└── src/                         # Python utilities
    ├── predict.py              # Prediction logic
    ├── preprocess.py           # Text preprocessing
    └── ...
```

## 🧪 Testing

### Test the ML Server
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"headline": "This one trick will shock you!"}'
```

### Expected Response
```json
{
  "headline": "This one trick will shock you!",
  "is_clickbait": true,
  "confidence": 0.95,
  "models_available": true
}
```

## 🐛 Troubleshooting

### Issue: ML Server Not Found
**Solution:** Make sure `python ml_server.py` is running on port 5000

### Issue: Models Not Loaded
**Solution:** Verify files exist:
- `models/clickbait_model.h5`
- `models/tokenizer.pkl`

### Issue: Port Already in Use
**Solution:** Change port in ml_server.py or kill existing process:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Issue: Dependencies Missing
**Solution:** Install required packages:
```bash
pip install -r requirements.txt
```

## 📈 Performance

- **Single Prediction:** ~50-100ms
- **Batch (100 headlines):** ~5-10 seconds
- **Server Startup:** ~2-3 seconds
- **Memory Usage:** ~500MB

## 🔐 Security

- CORS enabled for localhost only
- No data stored on server
- Models run locally
- No external API calls
- Input validation on all endpoints

## 🚀 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 ml_server:app
```

### Using Docker
```bash
docker build -t declickify .
docker run -p 5000:5000 declickify
```

### Frontend Production Build
```bash
cd frontend
npm run build
# Deploy 'dist' folder to web server
```

## 📝 CSV Format for Batch Upload

**Required:** Column named `headline`

Example:
```csv
headline
This one trick will change your life
Scientists discover new species
Breaking news about celebrity
Local man helps neighbor
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - Feel free to use for personal and commercial projects

## 📧 Support

For issues or questions:
1. Check [GETTING_STARTED.md](GETTING_STARTED.md)
2. Review [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
3. Check error messages in browser console
4. Verify both servers are running

## 🎓 How It Works

1. **User Input** → Headline text
2. **Preprocessing** → Clean and normalize text
3. **Tokenization** → Convert words to numbers
4. **Padding** → Ensure consistent input length
5. **Model Inference** → Neural network prediction
6. **Output** → Clickbait probability (0-1)
7. **Display** → Visual feedback to user

## 🌟 Key Highlights

✅ No external API calls - Everything runs locally
✅ Fast inference - Real-time predictions
✅ High accuracy - Deep learning model
✅ Easy to use - Simple, intuitive UI
✅ Batch capable - Process multiple headlines
✅ Open source - Available for modification

## 🔄 What's New

**Latest Update:** ML Model Integrated Directly
- Removed complex API framework
- Lightweight Flask server for inference
- Direct model access from frontend
- Simplified codebase
- Better performance

## 📊 Statistics

- **Model Accuracy:** 90%+
- **Training Data:** 5,000+ headlines
- **Inference Speed:** <100ms per headline
- **Supported Formats:** 5+ file types
- **Active Users:** Ready for deployment

---

**Ready to detect clickbait?** Start with [GETTING_STARTED.md](GETTING_STARTED.md) 🚀
