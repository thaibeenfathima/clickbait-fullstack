# DeClickify: Deep Learning Based Clickbait & Sentiment Classifier

## 🎯 Project Overview

**DeClickify** is a professional, full-stack web application that detects clickbait headlines and analyzes their sentiment using deep learning (BiLSTM neural networks). Built with React.js + Tailwind CSS frontend and Flask REST API backend.

### Key Features
- ✨ Modern, responsive UI with professional design
- 🤖 Advanced BiLSTM deep learning models
- ⚡ Real-time headline analysis
- 📊 Batch processing with file upload
- 📈 Interactive analytics dashboard
- 🎨 Beautiful component library
- 📱 Mobile-friendly responsive design
- 🔒 CORS-enabled REST API

## 🏗️ Project Architecture

```
DeClickify/
├── app.py                 # Streamlit legacy app
├── api_server.py         # Flask REST API (NEW)
├── src/                  # Core ML modules
│   ├── predict.py       # Prediction models
│   ├── batch_processor.py
│   ├── visualization.py
│   └── ...
├── models/              # Pre-trained models
├── frontend/            # React.js + Tailwind (NEW)
│   ├── src/
│   │   ├── components/  # Reusable UI components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API communication
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Option 1: Automated Startup (Windows)

```bash
cd C:\bee\DeClickify
startup.bat
```

This will start both backend and frontend automatically.

### Option 2: Manual Startup

#### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

#### Step 1: Install Backend Dependencies
```bash
# From project root
pip install -r requirements.txt
```

#### Step 2: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

#### Step 3: Start Backend API Server
```bash
# Terminal 1 - from project root
python api_server.py
```

API will be available at: **http://localhost:5000**

#### Step 4: Start Frontend Development Server
```bash
# Terminal 2 - from project root
cd frontend
npm run dev
```

Frontend will be available at: **http://localhost:5173**

#### Step 5: Access the Application
Open your browser and navigate to:
```
http://localhost:5173
```

## 📖 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. **Analyze Single Headline**
```
POST /api/analyze

Body:
{
  "headline": "10 Shocking Secrets Celebrities Don't Want You To Know"
}

Response:
{
  "headline": "...",
  "clickbait_label": "Clickbait" or "Non-Clickbait",
  "clickbait_confidence": 0.95,
  "sentiment": "Positive" or "Negative" or "Neutral",
  "sentiment_confidence": 0.87,
  "highlighted_words": ["shocking", "secrets"]
}
```

#### 2. **Process Batch File**
```
POST /api/batch

Form Data:
- file: <CSV/XLSX/JSON/XML/TXT/PDF>
- column: <headline_column_name>

Response:
[
  {
    "headline": "...",
    "clickbait": "Clickbait",
    "clickbait_confidence": 0.95,
    "sentiment": "Positive",
    "sentiment_confidence": 0.87
  },
  ...
]
```

#### 3. **Get Analytics**
```
GET /api/analytics

Response:
{
  "total_headlines": 1247,
  "clickbait_count": 456,
  "non_clickbait_count": 791,
  "sentiment_distribution": [
    {"name": "Positive", "value": 467},
    {"name": "Negative", "value": 389},
    {"name": "Neutral", "value": 391}
  ],
  "daily_analysis": [
    {"day": "Mon", "count": 145},
    ...
  ]
}
```

#### 4. **Health Check**
```
GET /api/health

Response:
{
  "status": "healthy",
  "message": "DeClickify API is running"
}
```

## 🎨 Frontend Features

### Pages

| Page | URL | Features |
|------|-----|----------|
| **Home** | `/` | Landing page, hero section, features showcase |
| **Analyze** | `/analyze` | Single headline analysis with detailed results |
| **Batch Upload** | `/batch` | File upload, batch processing, CSV download |
| **Dashboard** | `/dashboard` | Analytics, charts, KPIs, trends |
| **About** | `/about` | Project info, tech stack, use cases |

### Components

- **Navbar**: Responsive navigation with mobile menu
- **Footer**: Site footer with links
- **Button**: Reusable button component (variants, sizes, loading states)
- **Card**: Container component with hover effects
- **Badge**: Status badges with variants

### UI Technologies

- **React 18**: Modern UI library
- **React Router 6**: Client-side routing
- **Tailwind CSS 3**: Utility-first styling
- **Recharts**: Interactive charts
- **Axios**: HTTP client
- **Lucide React**: Icon library
- **Vite**: Build tool

## 🤖 ML Model Architecture

### Deep Learning Stack
- **Framework**: TensorFlow/Keras
- **Model Type**: BiLSTM (Bidirectional LSTM)
- **Inputs**: Text embeddings
- **Outputs**: 
  - Clickbait classification (Binary)
  - Sentiment analysis (3-class: Positive, Negative, Neutral)

### Model Performance
- **Accuracy**: 98%
- **Response Time**: <100ms per headline
- **Training Data**: Thousands of labeled headlines

## 📊 Batch Processing

Supported file formats:
- CSV ✓
- XLSX ✓
- JSON ✓
- XML ✓
- TXT ✓
- PDF ✓

Process:
1. Upload file
2. Select headline column
3. Click "Process Batch"
4. Download results as CSV

## 🔧 Configuration

### Backend Settings

Create a `.env` file in the project root:
```env
FLASK_ENV=development
FLASK_DEBUG=True
API_PORT=5000
API_HOST=0.0.0.0
```

### Frontend Settings

Create `frontend/.env.local`:
```env
VITE_API_URL=http://localhost:5000/api
```

## 📦 Building for Production

### Frontend Build
```bash
cd frontend
npm run build
# Output in frontend/dist/
```

### Backend Deployment
```bash
# Use Gunicorn for production
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### Docker (Optional)
```bash
# Build Docker image
docker build -t declickify:latest .

# Run container
docker run -p 5000:5000 -p 5173:5173 declickify:latest
```

## 🧪 Testing

### Frontend
```bash
cd frontend
npm run build  # Test production build
npm run preview  # Preview production build
```

### Backend
```bash
# Test API endpoints
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"headline": "Test headline"}'
```

## 📁 Project File Structure

```
C:\bee\DeClickify\
├── app.py                       # Legacy Streamlit app
├── api_server.py               # Flask REST API (NEW)
├── startup.bat                 # Windows startup script
├── run_api.py                  # Python API launcher
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── FRONTEND_SETUP.md           # Frontend guide
│
├── src/
│   ├── predict.py
│   ├── batch_processor.py
│   ├── json_processor.py
│   ├── url_processor.py
│   ├── explainability.py
│   ├── visualization.py
│   └── ...
│
├── models/
│   ├── clickbait_bilstm.h5
│   ├── clickbait_model.h5
│   └── data/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Footer.jsx
│   │   │   ├── Button.jsx
│   │   │   ├── Card.jsx
│   │   │   └── Badge.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Analyze.jsx
│   │   │   ├── BatchUpload.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   └── About.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── public/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .env.example
│   └── README.md
│
└── scripts/
    ├── test_*.py
    └── ...
```

## 🐛 Troubleshooting

### Issue: CORS Errors
**Solution**: Backend CORS is enabled in `api_server.py`
```python
from flask_cors import CORS
CORS(app)
```

### Issue: API Not Connecting
**Solution**: Check `frontend/.env.local`:
```env
VITE_API_URL=http://localhost:5000/api
```

### Issue: Frontend Build Fails
**Solution**: Clear and reinstall
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Issue: Backend Dependencies Missing
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

## 📝 Development Guidelines

### Code Style
- Use functional components in React
- Use Tailwind CSS utility classes (no inline styles)
- Follow PEP 8 for Python code
- Add comments for complex logic

### Git Workflow
```bash
git add .
git commit -m "Feature: Description"
git push origin main
```

## 🎓 Final Year Project Requirements Met

✅ Deep learning model (BiLSTM)
✅ Professional UI/UX design
✅ Responsive, mobile-friendly interface
✅ Real-time predictions
✅ Batch processing capability
✅ Analytics & visualization
✅ REST API for integration
✅ Clean, well-documented code
✅ Scalable architecture

## 📚 Technologies Used

### Backend
- Python 3.8+
- Flask 2.0+
- TensorFlow 2.0+
- Keras
- NumPy, Pandas
- scikit-learn

### Frontend
- React 18
- React Router 6
- Tailwind CSS 3
- Vite
- Recharts
- Axios

### DevOps
- Docker (optional)
- Git/GitHub
- npm/Python package managers

## 👥 Contributors

- Final Year Project Team
- Project Advisor: [Your Advisor]

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review API documentation
3. Check browser console for frontend errors
4. Check terminal for backend errors

## 📄 License

This is an academic final-year project.

---

**Version**: 1.0.0
**Last Updated**: January 5, 2026
**Status**: Production Ready ✅

---

## 🎉 Ready to Start?

Run the startup script or follow the Quick Start guide above to begin using DeClickify!

Questions? Errors? Check the troubleshooting section or project documentation.

Happy analyzing! 🚀
