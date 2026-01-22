# Panchayat - Sentiment Analysis & Trend Detection

A full-stack system for real-time sentiment analysis and trend detection on Reddit data.

## Tech Stack

- **Backend**: Python, FastAPI, HuggingFace Transformers, BERTopic
- **Frontend**: React, Vite, Tailwind CSS, Recharts
- **Data**: Reddit API (PRAW), SQLite

## Project Structure

```
panchayat/
├── backend/           # FastAPI + ML Pipeline
│   ├── models/        # Sentiment models (BERT, LSTM, RF)
│   ├── nlp/           # Topic modeling & trends
│   ├── data/          # Reddit client & database
│   └── api/           # REST endpoints
└── frontend/          # React Dashboard
    └── src/
        └── components/
```

## Quick Start

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Credits / Mix-Match Sources

| Component | Source Repository |
|-----------|------------------|
| BERT Sentiment | HuggingFace `transformers` |
| Topic Modeling | `MaartenGr/BERTopic` |
| Dashboard Template | `TailAdmin/free-react-tailwind-admin-dashboard` |
| Reddit API | `praw-dev/praw` |
