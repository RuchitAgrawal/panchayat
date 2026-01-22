"""
Panchayat - Sentiment Analysis & Trend Detection API

FastAPI backend for analyzing Reddit posts and detecting trends.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Panchayat API",
    description="Sentiment Analysis & Trend Detection for Reddit",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "project": "Panchayat",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": False,  # Will be True once models are initialized
        "reddit_configured": bool(settings.reddit_client_id),
        "default_subreddits": settings.default_subreddits
    }


# Placeholder endpoints - will be implemented in Phase 2
@app.post("/api/analyze")
async def analyze_text(text: str):
    """
    Analyze sentiment of a single text.
    TODO: Implement ensemble model prediction.
    """
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "sentiment": "neutral",
        "confidence": 0.0,
        "message": "Model not loaded yet - Phase 2"
    }


@app.get("/api/trends")
async def get_trends():
    """
    Get sentiment trends over time.
    TODO: Implement trend aggregation.
    """
    return {
        "trends": [],
        "message": "Not implemented yet - Phase 3"
    }


@app.get("/api/topics")
async def get_topics():
    """
    Get trending topics.
    TODO: Implement BERTopic.
    """
    return {
        "topics": [],
        "message": "Not implemented yet - Phase 3"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
