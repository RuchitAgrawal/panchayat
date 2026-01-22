# 🏛️ Panchayat - Sentiment Analysis Dashboard

A full-stack sentiment analysis platform that combines advanced ML models with a modern React dashboard for real-time insights.

![Dashboard Preview](docs/dashboard-dark.png)

## ✨ Features

- **🤖 ML Ensemble** - BERT, LSTM (TextBlob), and Random Forest with weighted voting
- **📊 Real-time Dashboard** - Clean React UI with light/dark theme toggle
- **📈 Trend Analysis** - Time-series sentiment tracking with Recharts
- **🔄 Multiple Data Sources** - Reddit API, CSV import, sample data
- **💾 SQLite Storage** - Persistent post storage with sentiment scores
- **📱 Responsive Design** - Works on desktop and mobile

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Backend Setup

```bash
cd backend
python -m venv venv
./venv/Scripts/activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Open the App
- **Dashboard:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs

## 📂 Project Structure

```
panchayat/
├── backend/
│   ├── main.py              # FastAPI app with all endpoints
│   ├── config.py            # Configuration settings
│   ├── models/              # ML models (BERT, LSTM, RF, Ensemble)
│   ├── nlp/                 # Topic modeling, trends, N-grams
│   ├── data/                # Reddit client, database, sample data
│   └── datasets/            # Place Kaggle CSVs here
│
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main dashboard
    │   ├── components/      # React components
    │   ├── hooks/           # Theme context
    │   └── api/             # Backend API client
    └── index.css            # Theme system
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Analyze text sentiment |
| `/api/analyze/batch` | POST | Batch analysis |
| `/api/trends` | GET | Get sentiment trends |
| `/api/posts` | GET | Get stored posts |
| `/api/posts/stats` | GET | Sentiment statistics |
| `/api/sample/quick` | GET | Load sample data |
| `/api/kaggle/sentiment140` | POST | Load Kaggle dataset |

## 📊 ML Models

| Model | Weight | Source |
|-------|--------|--------|
| BERT | 50% | `nlptown/bert-base-multilingual-uncased-sentiment` |
| LSTM | 30% | TextBlob (fallback) |
| Random Forest | 20% | TF-IDF + sklearn |

## 🎨 Dashboard Features

- **Theme Toggle** - Light ↔ Dark mode
- **Sentiment Gauge** - Overall score visualization
- **Trend Chart** - Time-series analysis
- **Stats Cards** - Post counts & percentages
- **Recent Posts** - Table with sentiment badges

## 📁 Using Kaggle Datasets

1. Download [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. Place CSV in `backend/datasets/sentiment140.csv`
3. Load via API:
```bash
curl -X POST http://localhost:8000/api/kaggle/sentiment140 \
  -H "Content-Type: application/json" \
  -d '{"limit": 200, "balanced": true}'
```

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- PyTorch + Transformers (BERT)
- scikit-learn (Random Forest)
- SQLAlchemy + SQLite
- PRAW (Reddit API)

**Frontend:**
- React 18 + Vite
- Recharts
- CSS Variables (theming)

## 📝 License

MIT License - feel free to use and modify!

---

Built with ❤️ using Python, React, and ML
