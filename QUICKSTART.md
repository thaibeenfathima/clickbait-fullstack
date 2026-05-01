## 🚀 DECLICKIFY - QUICK START GUIDE

### ⚡ 60-Second Setup

```bash
# 1. Open Terminal
cd C:\bee\DeClickify

# 2. Install Backend Dependencies (one-time)
pip install -r requirements.txt

# 3. Open TWO Terminals

# TERMINAL 1: Backend API
python api_server.py
# Should see: Running on http://localhost:5000

# TERMINAL 2: Frontend
cd frontend && npm run dev
# Should see: Ready at http://localhost:5173
```

### 📱 Open in Browser
```
http://localhost:5173
```

---

## 🎯 What You Get

| Feature | Status | Details |
|---------|--------|---------|
| **Home Page** | ✅ Complete | Landing page with features |
| **Analyze Page** | ✅ Complete | Single headline analysis |
| **Batch Upload** | ✅ Complete | CSV/XLSX file processing |
| **Dashboard** | ✅ Complete | Analytics & charts |
| **About Page** | ✅ Complete | Project information |
| **Responsive Design** | ✅ Complete | Mobile, tablet, desktop |
| **API Integration** | ✅ Complete | Ready for backend |
| **Professional UI** | ✅ Complete | Modern Tailwind design |

---

## 📂 Key Files

### Frontend (React.js + Tailwind)
- `frontend/src/App.jsx` - Main app & routing
- `frontend/src/pages/` - All 5 pages
- `frontend/src/components/` - Reusable components
- `frontend/src/services/api.js` - API communication

### Backend (Flask API)
- `api_server.py` - REST API wrapper ⭐NEW
- `requirements.txt` - Dependencies ⭐UPDATED

### Documentation
- `PROJECT_README.md` - Complete guide ⭐NEW
- `IMPLEMENTATION_SUMMARY.md` - What was built ⭐NEW
- `VISUAL_GUIDE.md` - Visual reference ⭐NEW
- `FRONTEND_SETUP.md` - Setup details ⭐NEW

---

## 🔗 Important URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:5000 |
| API Health | http://localhost:5000/api/health |

---

## 🎨 Pages

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | Landing page |
| Analyze | `/analyze` | Single analysis |
| Batch | `/batch` | Bulk upload |
| Dashboard | `/dashboard` | Analytics |
| About | `/about` | Project info |

---

## 🔌 API Endpoints

```
POST /api/analyze
Body: {"headline": "..."}
Response: {clickbait_label, confidence, sentiment, ...}

POST /api/batch
Body: File + column name
Response: Array of results

GET /api/analytics
Response: Dashboard data

GET /api/health
Response: Status check
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port 5000 in use** | Change port in `api_server.py` |
| **Port 5173 in use** | Vite will auto-increment |
| **API not connecting** | Check `VITE_API_URL` in `frontend/.env.local` |
| **Module not found** | Run `npm install` in frontend folder |
| **Build fails** | Delete `node_modules` & reinstall |
| **Python imports fail** | Run `pip install -r requirements.txt` |

---

## 📝 Environment Setup

### Frontend (.env.local)
```env
VITE_API_URL=http://localhost:5000/api
```

### Backend (optional .env)
```env
FLASK_ENV=development
FLASK_DEBUG=True
API_PORT=5000
```

---

## 🏗️ Tech Stack

**Frontend:**
- React 18
- React Router 6
- Tailwind CSS 3
- Recharts
- Axios
- Vite

**Backend:**
- Flask
- TensorFlow
- Keras
- NumPy
- Pandas

---

## ✨ Features Highlights

### Home Page
- Hero section
- Feature cards
- Stats display
- CTAs

### Analyze Page
- Text input
- Real-time results
- Confidence bars
- Highlighted keywords

### Batch Page
- File upload
- Preview table
- Results download
- CSV export

### Dashboard
- KPI cards
- Bar charts
- Pie charts
- Activity feed

### About Page
- Project info
- Tech details
- Use cases
- Contact

---

## 🎓 For Your Presentation

**Features to Highlight:**
1. Professional UI design
2. Real-time ML predictions
3. Batch processing capability
4. Interactive analytics
5. Responsive mobile design
6. REST API integration
7. Production-ready code
8. Clean architecture

**Live Demo:**
1. Show home page
2. Analyze a clickbait headline
3. Upload batch CSV
4. Show dashboard analytics
5. Explain architecture

---

## 📞 Support

**Documentation:**
- `PROJECT_README.md` - Full guide
- `IMPLEMENTATION_SUMMARY.md` - What's included
- `VISUAL_GUIDE.md` - Architecture & flow
- `FRONTEND_SETUP.md` - Frontend details

**Code Comments:**
- Check component files for implementation details
- API service has request/response examples

---

## 🎉 You're Ready!

Everything is set up and ready to run.
Just follow the 60-second setup above.

**Status:** ✅ Production Ready

Questions? Check the documentation files.
