# DeClickify - Professional Frontend Setup

## 📋 Overview

This is a **production-ready React.js + Tailwind CSS** frontend for **DeClickify**, a deep learning-based clickbait detection and sentiment classification system.

## 🎯 Features

- ✅ Modern, responsive UI with Tailwind CSS
- ✅ React Router for navigation
- ✅ Real-time API integration with Axios
- ✅ Interactive charts with Recharts
- ✅ Professional components library
- ✅ Mobile-friendly design
- ✅ Accessibility compliant
- ✅ Production-optimized with Vite

## 📁 Folder Structure

```
src/
├── components/        # Reusable UI components
├── pages/            # Page components
├── services/         # API communication
├── utils/            # Helper functions
├── App.jsx          # Main app component
├── index.css        # Global styles + Tailwind
└── main.jsx         # Entry point
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Setup Environment
```bash
cp .env.example .env.local
```

### 3. Run Development Server
```bash
npm run dev
```

Visit `http://localhost:5173`

### 4. Build for Production
```bash
npm run build
```

## 📖 Pages

| Page | Path | Features |
|------|------|----------|
| **Home** | `/` | Landing page, hero section, features |
| **Analyze** | `/analyze` | Single headline analysis, results |
| **Batch Upload** | `/batch` | File upload, bulk processing |
| **Dashboard** | `/dashboard` | Analytics, charts, KPIs |
| **About** | `/about` | Project info, tech stack |

## 🔌 API Endpoints

### Analyze Headline
```
POST /api/analyze
```

### Process Batch
```
POST /api/batch
```

### Get Analytics
```
GET /api/analytics
```

## 🛠 Tech Stack

- React 19
- React Router 6
- Tailwind CSS 3
- Recharts
- Axios
- Lucide React
- Vite

## 📱 Responsive Design

- Mobile-first approach
- Tablet optimization
- Desktop layouts
- Hamburger menu on mobile

## 🎨 Styling

All styling uses **Tailwind CSS** utility classes.

Color scheme:
- Primary: Blue
- Success: Green
- Danger: Red
- Neutral: Gray

## 🔑 Environment Variables

```env
VITE_API_URL=http://localhost:5000/api
```

## 💡 Component Examples

### Button
```jsx
<Button variant="primary" isLoading={loading}>
  Analyze
</Button>
```

### Card
```jsx
<Card className="p-6">Content</Card>
```

### Badge
```jsx
<Badge label="Clickbait" variant="danger" />
```

## 🐛 Troubleshooting

**CORS errors?** → Check backend CORS configuration
**API not connecting?** → Verify `VITE_API_URL` in `.env.local`
**Build fails?** → Run `npm install` again

## 📝 License

Academic project - Final Year