"""
Panchayat - Sentiment Analysis & Trend Detection API

FastAPI backend for analyzing Reddit posts and detecting trends.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Lazy load ensemble model
_ensemble = None


def get_ensemble():
    """Lazy load the ensemble model."""
    global _ensemble
    if _ensemble is None:
        from models.ensemble import SentimentEnsemble
        logger.info("Initializing sentiment ensemble...")
        _ensemble = SentimentEnsemble()
        logger.info("Ensemble initialized successfully")
    return _ensemble


# Request/Response Models
class AnalyzeRequest(BaseModel):
    text: str
    include_breakdown: bool = True


class BatchAnalyzeRequest(BaseModel):
    texts: List[str]
    include_breakdown: bool = False


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
        "models_loaded": _ensemble is not None,
        "reddit_configured": bool(settings.reddit_client_id),
        "default_subreddits": settings.default_subreddits
    }


@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze sentiment of a single text using ensemble model.
    
    Returns:
        - label: positive/negative/neutral
        - confidence: 0-1 confidence score
        - score: -1 to +1 numeric score
        - breakdown: individual model predictions (optional)
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        ensemble = get_ensemble()
        result = ensemble.predict(request.text, include_breakdown=request.include_breakdown)
        
        return {
            "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            **result
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/batch")
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Analyze sentiment of multiple texts.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        ensemble = get_ensemble()
        results = ensemble.predict_batch(request.texts, include_breakdown=request.include_breakdown)
        
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze/quick")
async def quick_analyze(text: str = Query(..., min_length=1)):
    """
    Quick sentiment analysis (GET request for easy testing).
    """
    try:
        ensemble = get_ensemble()
        result = ensemble.predict(text, include_breakdown=False)
        return result
    except Exception as e:
        logger.error(f"Quick analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Placeholder endpoints - will be implemented in Phase 3
@app.get("/api/trends")
async def get_trends():
    """
    Get sentiment trends over time.
    TODO: Implement in Phase 3.
    """
    return {
        "trends": [],
        "message": "Not implemented yet - Phase 3"
    }


@app.get("/api/topics")
async def get_topics():
    """
    Get trending topics.
    TODO: Implement in Phase 3.
    """
    return {
        "topics": [],
        "message": "Not implemented yet - Phase 3"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
