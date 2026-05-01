# 🚀 Quick Reference Card - DeClickify

## Start in 30 Seconds

```bash
# Terminal 1
python ml_server.py

# Terminal 2
cd frontend && npm install && npm run dev

# Browser
Open: http://localhost:5173
```

## Commands Cheat Sheet

| Task | Command |
|------|---------|
| Start ML Server | `python ml_server.py` |
| Start Frontend | `cd frontend && npm run dev` |
| Clean up old files | `cleanup_old_files.bat` |
| Build frontend | `cd frontend && npm run build` |
| Check ML status | `curl http://localhost:5000/api/status` |
| Install frontend deps | `cd frontend && npm install` |

## Ports Reference

| Service | Port | URL |
|---------|------|-----|
| ML Server | 5000 | http://localhost:5000 |
| Frontend | 5173 | http://localhost:5173 |
| API Endpoint | 5000 | http://localhost:5000/api |

## File Locations

| Item | Location |
|------|----------|
| ML Server | `ml_server.py` |
| Frontend | `frontend/` |
| Models | `models/` |
| Frontend Pages | `frontend/src/pages/` |
| API Client | `frontend/src/services/api.js` |

## API Quick Reference

### Analyze Headline
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"headline":"Example headline"}'
```

### Check Server
```bash
curl http://localhost:5000/api/status
```

### Batch Upload (from frontend)
1. Go to Batch Upload page
2. Select CSV file with `headline` column
3. Click Process

## Frontend Features

| Feature | Location | How To Use |
|---------|----------|-----------|
| Single Analysis | Analyze Page | Enter text + Click Analyze |
| Batch Processing | Batch Upload | Upload CSV + Click Process |
| Results Download | Batch Results | Click Download CSV |
| Status Check | Any page (in error) | Check browser console |

## Model Information

- **Type:** BiLSTM Neural Network
- **Model File:** `models/clickbait_model.h5`
- **Tokenizer:** `models/tokenizer.pkl`
- **Accuracy:** 90%+
- **Speed:** <100ms per prediction
- **Input:** Text headlines
- **Output:** Clickbait probability (0.0-1.0)

## Response Format

```json
{
  "headline": "...",
  "is_clickbait": true,
  "confidence": 0.92,
  "models_available": true,
  "prediction": {
    "clickbait_score": 0.92,
    "non_clickbait_score": 0.08
  }
}
```

## Batch Response Format

```json
{
  "results": [
    {
      "index": 0,
      "headline": "...",
      "is_clickbait": true,
      "confidence": 0.92
    }
  ],
  "total": 100,
  "models_available": true
}
```

## Troubleshooting Quick Fixes

| Problem | Fix |
|---------|-----|
| ML Server won't start | `python ml_server.py` or check port 5000 |
| Models not found | Check `models/clickbait_model.h5` exists |
| Port in use | Kill process on port 5000 |
| Frontend error | Make sure ML server running on 5000 |
| Old API still called | Update `VITE_ML_SERVER_URL` env var |

## Environment Variables

```env
# frontend/.env.local
VITE_ML_SERVER_URL=http://localhost:5000/api
```

## Key Files Modified

```
✅ ml_server.py (NEW)
✅ frontend/src/services/api.js (UPDATED)
✅ frontend/src/pages/Analyze.jsx (UPDATED)
✅ frontend/src/pages/BatchUpload.jsx (UPDATED)
✅ frontend/package.json (UPDATED)
```

## Key Files to Remove

```
Run: cleanup_old_files.bat

❌ api_server.py
❌ run_api.py
❌ app.py
❌ import_*.py
```

## Learning Path

1. Start: [GETTING_STARTED.md](GETTING_STARTED.md)
2. Architecture: [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
3. Frontend: [FRONTEND_STARTUP_GUIDE.md](FRONTEND_STARTUP_GUIDE.md)
4. Technical: [ML_INTEGRATION_SUMMARY.md](ML_INTEGRATION_SUMMARY.md)

## Browser DevTools Tips

**Check API Calls:**
1. Open DevTools (F12)
2. Go to Network tab
3. Submit headline
4. See POST request to `localhost:5000/api/analyze`
5. Check Response tab for JSON result

## Common Issues & Solutions

### ❌ "Cannot POST /api/analyze"
**Solution:** ML server not running - run `python ml_server.py`

### ❌ "Models not loaded"
**Solution:** Check `models/clickbait_model.h5` and `models/tokenizer.pkl` exist

### ❌ "Address already in use"
**Solution:** Port 5000 in use - kill process or use different port

### ❌ "Cannot GET /api/analyze"
**Solution:** Frontend calling old API - restart server and frontend

## Performance Tips

- ML server starts in ~2-3 seconds
- Single prediction: ~50-100ms
- Batch of 100: ~5-10 seconds
- Memory: ~500MB for TensorFlow

## Testing the System

```bash
# 1. Check ML server running
curl http://localhost:5000/api/status

# 2. Test prediction
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"headline":"This will shock you"}'

# 3. Check frontend running
# Open http://localhost:5173 in browser

# 4. Test through UI
# - Enter headline
# - Click Analyze
# - See results
```

## Production Checklist

- [ ] Run `cleanup_old_files.bat`
- [ ] Test ML server: `python ml_server.py`
- [ ] Test frontend: `npm run dev`
- [ ] Build frontend: `npm run build`
- [ ] Set up ML server with gunicorn
- [ ] Deploy to production server
- [ ] Test all features
- [ ] Monitor error logs

## Quick Reference URLs

| URL | Purpose |
|-----|---------|
| http://localhost:5173 | Frontend app |
| http://localhost:5000 | ML server |
| http://localhost:5000/api/status | Check ML status |
| http://localhost:5000/api/analyze | Analyze endpoint |
| http://localhost:5000/api/batch | Batch endpoint |

## Version Info

- **Frontend:** React 18 + Vite
- **Backend:** Flask + TensorFlow/Keras
- **Models:** Trained BiLSTM network
- **Python:** 3.8+
- **Node:** 16+

---

**Need help?** Check the documentation files in the project root! 📚
