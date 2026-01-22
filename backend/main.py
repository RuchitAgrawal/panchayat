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


# ===== Phase 3: Topic Modeling & Trends =====

# Lazy load NLP modules
_trend_detector = None
_ngram_analyzer = None
_topic_modeler = None


def get_trend_detector():
    """Lazy load trend detector."""
    global _trend_detector
    if _trend_detector is None:
        from nlp.trend_detector import TrendDetector
        _trend_detector = TrendDetector()
    return _trend_detector


def get_ngram_analyzer():
    """Lazy load N-gram analyzer."""
    global _ngram_analyzer
    if _ngram_analyzer is None:
        from nlp.ngram_analyzer import NgramAnalyzer
        _ngram_analyzer = NgramAnalyzer()
    return _ngram_analyzer


def get_topic_modeler():
    """Lazy load topic modeler."""
    global _topic_modeler
    if _topic_modeler is None:
        from nlp.topic_modeling import TopicModeler
        logger.info("Initializing TopicModeler (this may take a moment)...")
        _topic_modeler = TopicModeler()
        logger.info("TopicModeler initialized")
    return _topic_modeler


# Request models for Phase 3
class TopicExtractionRequest(BaseModel):
    texts: List[str]
    top_n_words: int = 10


class NgramExtractionRequest(BaseModel):
    texts: List[str]
    top_k: int = 20
    ngram_type: str = "all"  # "unigrams", "bigrams", "trigrams", "all"


class AnalyzeAndTrackRequest(BaseModel):
    text: str
    source: Optional[str] = None
    include_breakdown: bool = False


@app.get("/api/trends")
async def get_trends(period: str = "1h", limit: int = 24):
    """
    Get aggregated sentiment trends over time.
    
    Args:
        period: Aggregation period ('1h', '6h', '1d', '1w')
        limit: Maximum time buckets to return
    
    Returns:
        List of time-bucketed sentiment averages
    """
    try:
        detector = get_trend_detector()
        trends = detector.get_trends(period=period, limit=limit)
        summary = detector.get_summary()
        
        return {
            "trends": trends,
            "summary": summary,
            "period": period
        }
    except Exception as e:
        logger.error(f"Trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trends/recent")
async def get_recent_entries(limit: int = 20):
    """Get most recent analyzed entries."""
    try:
        detector = get_trend_detector()
        return {
            "entries": detector.get_recent(limit=limit),
            "total_tracked": len(detector.history)
        }
    except Exception as e:
        logger.error(f"Recent entries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/track")
async def analyze_and_track(request: AnalyzeAndTrackRequest):
    """
    Analyze text AND track it for trend detection.
    
    This endpoint both returns the sentiment and stores it
    in the trend detector for time-series analysis.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Analyze sentiment
        ensemble = get_ensemble()
        result = ensemble.predict(request.text, include_breakdown=request.include_breakdown)
        
        # Track for trends
        detector = get_trend_detector()
        detector.add_entry(
            text=request.text,
            sentiment=result,
            source=request.source
        )
        
        return {
            "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "tracked": True,
            "total_tracked": len(detector.history),
            **result
        }
    except Exception as e:
        logger.error(f"Analyze+track error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/topics")
async def get_topics():
    """
    Get current topic summary from tracked entries.
    
    Uses N-grams for a lightweight topic representation.
    For full BERTopic extraction, use POST /api/topics/extract.
    """
    try:
        detector = get_trend_detector()
        if not detector.history:
            return {
                "topics": [],
                "message": "No entries tracked yet. Use /api/analyze/track to add entries."
            }
        
        # Extract texts from history
        texts = [e["text"] for e in detector.history]
        
        # Use N-gram analyzer for quick topic keywords
        analyzer = get_ngram_analyzer()
        keywords = analyzer.extract_ngrams(texts, top_k=30, ngram_range=(1, 2))
        
        return {
            "keywords": keywords,
            "word_cloud": analyzer.get_word_cloud_data(texts, max_words=50),
            "total_documents": len(texts)
        }
    except Exception as e:
        logger.error(f"Topics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/topics/extract")
async def extract_topics(request: TopicExtractionRequest):
    """
    Extract semantic topics using BERTopic.
    
    This uses BERT embeddings to cluster similar documents
    and extract representative topic keywords.
    
    Note: Requires at least 3 documents for meaningful clustering.
    """
    if not request.texts or len(request.texts) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Need at least 3 texts for topic extraction"
        )
    
    try:
        modeler = get_topic_modeler()
        result = modeler.extract_topics(request.texts, top_n_words=request.top_n_words)
        
        return result
    except Exception as e:
        logger.error(f"Topic extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ngrams")
async def extract_ngrams(request: NgramExtractionRequest):
    """
    Extract frequent N-grams from texts.
    
    Args:
        texts: List of documents
        top_k: Number of top N-grams to return
        ngram_type: 'unigrams', 'bigrams', 'trigrams', or 'all'
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        analyzer = get_ngram_analyzer()
        
        if request.ngram_type == "unigrams":
            ngrams = analyzer.extract_unigrams(request.texts, top_k=request.top_k)
        elif request.ngram_type == "bigrams":
            ngrams = analyzer.extract_bigrams(request.texts, top_k=request.top_k)
        elif request.ngram_type == "trigrams":
            ngrams = analyzer.extract_trigrams(request.texts, top_k=request.top_k)
        else:
            ngrams = analyzer.extract_ngrams(request.texts, top_k=request.top_k)
        
        return {
            "ngrams": ngrams,
            "type": request.ngram_type,
            "document_count": len(request.texts)
        }
    except Exception as e:
        logger.error(f"N-gram extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/wordcloud")
async def get_wordcloud_data():
    """
    Get word cloud data from tracked entries.
    
    Returns data formatted for react-wordcloud component.
    """
    try:
        detector = get_trend_detector()
        if not detector.history:
            return {"words": [], "message": "No entries tracked yet"}
        
        texts = [e["text"] for e in detector.history]
        analyzer = get_ngram_analyzer()
        
        return {
            "words": analyzer.get_word_cloud_data(texts, max_words=100)
        }
    except Exception as e:
        logger.error(f"Word cloud error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Phase 4: Reddit Data Ingestion =====

# Lazy load Reddit modules
_collector = None
_database = None


def get_collector():
    """Lazy load batch collector."""
    global _collector
    if _collector is None:
        from data.batch_collector import BatchCollector
        _collector = BatchCollector()
    return _collector


def get_database():
    """Lazy load database."""
    global _database
    if _database is None:
        from data.database import get_db
        _database = get_db()
    return _database


# Request models for Phase 4
class CollectRequest(BaseModel):
    subreddit: str
    sort: str = "hot"
    limit: int = 25


class CollectMultipleRequest(BaseModel):
    subreddits: List[str]
    sort: str = "hot"
    limit: int = 10


@app.get("/api/reddit/status")
async def reddit_status():
    """Check if Reddit API is configured and working."""
    from data.reddit_client import is_reddit_configured
    
    return {
        "configured": is_reddit_configured(),
        "message": "Reddit API is ready" if is_reddit_configured() else "Reddit API not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env"
    }


@app.post("/api/collect")
async def collect_subreddit(request: CollectRequest):
    """
    Collect and analyze posts from a subreddit.
    
    Fetches posts, runs sentiment analysis, and stores in database.
    Also tracks to trend detector for real-time trends.
    """
    try:
        collector = get_collector()
        result = collector.collect_subreddit(
            subreddit=request.subreddit,
            sort=request.sort,
            limit=request.limit
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Collection failed"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collect/multiple")
async def collect_multiple_subreddits(request: CollectMultipleRequest):
    """Collect from multiple subreddits at once."""
    try:
        collector = get_collector()
        result = collector.collect_multiple(
            subreddits=request.subreddits,
            sort=request.sort,
            limit=request.limit
        )
        return result
    except Exception as e:
        logger.error(f"Multi-collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/posts")
async def get_posts(
    subreddit: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0
):
    """
    Get analyzed posts from database.
    
    Args:
        subreddit: Filter by subreddit (optional)
        limit: Maximum posts to return
        offset: Pagination offset
    """
    try:
        db = get_database()
        posts = db.get_posts(subreddit=subreddit, limit=limit, offset=offset)
        total = db.get_post_count(subreddit=subreddit)
        
        return {
            "posts": posts,
            "count": len(posts),
            "total": total,
            "subreddit": subreddit
        }
    except Exception as e:
        logger.error(f"Get posts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/posts/stats")
async def get_posts_stats(subreddit: Optional[str] = None):
    """Get sentiment statistics for stored posts."""
    try:
        db = get_database()
        stats = db.get_sentiment_stats(subreddit=subreddit)
        subreddits = db.get_subreddits()
        
        return {
            "stats": stats,
            "subreddits": subreddits,
            "filter": subreddit
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/subreddits")
async def get_subreddits():
    """Get list of all collected subreddits."""
    try:
        db = get_database()
        subreddits = db.get_subreddits()
        
        # Get stats for each
        subreddit_stats = []
        for sub in subreddits:
            stats = db.get_sentiment_stats(subreddit=sub)
            subreddit_stats.append({
                "name": sub,
                "post_count": stats.get("total", 0),
                "avg_sentiment": stats.get("avg_score", 0)
            })
        
        return {
            "subreddits": subreddit_stats,
            "total": len(subreddits)
        }
    except Exception as e:
        logger.error(f"Subreddits error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collection/stats")
async def get_collection_stats():
    """Get overall collection statistics."""
    try:
        collector = get_collector()
        stats = collector.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Collection stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Sample Data (for testing without Reddit API) =====

_sample_loader = None


def get_sample_loader():
    """Lazy load sample data loader."""
    global _sample_loader
    if _sample_loader is None:
        from data.sample_data import SampleDataLoader
        _sample_loader = SampleDataLoader()
    return _sample_loader


class LoadSampleRequest(BaseModel):
    count: int = 20
    category: Optional[str] = None
    analyze: bool = True


class LoadCSVRequest(BaseModel):
    filepath: str
    text_column: str = "text"
    title_column: Optional[str] = None
    limit: int = 100
    analyze: bool = True


@app.get("/api/sample/categories")
async def get_sample_categories():
    """Get available sample data categories."""
    loader = get_sample_loader()
    return {
        "categories": loader.get_categories(),
        "total_samples": len(loader.samples)
    }


@app.post("/api/sample/load")
async def load_sample_data(request: LoadSampleRequest):
    """
    Load and analyze built-in sample data.
    
    Use this when Reddit API is not configured.
    """
    try:
        loader = get_sample_loader()
        posts = loader.get_sample_posts(count=request.count, category=request.category)
        
        if not posts:
            return {"success": True, "message": "No matching samples found", "posts_loaded": 0}
        
        # Analyze and store
        db = get_database()
        collector = get_collector()
        ensemble = get_ensemble()
        
        analyzed = []
        for post in posts:
            # Combine title and text
            text = post["title"]
            if post.get("selftext"):
                text += " " + post["selftext"][:500]
            
            # Analyze sentiment
            if request.analyze:
                sentiment = ensemble.predict(text, include_breakdown=False)
            else:
                sentiment = {"label": "neutral", "score": 0, "confidence": 0}
            
            analyzed.append({
                "post": post,
                "sentiment": sentiment
            })
            
            # Track for trends
            collector.trend_detector.add_entry(
                text=text[:200],
                sentiment=sentiment,
                timestamp=post["created_utc"],
                source=post["subreddit"]
            )
        
        # Save to database
        saved = db.save_posts_batch(analyzed)
        
        return {
            "success": True,
            "posts_loaded": len(posts),
            "posts_analyzed": len(analyzed) if request.analyze else 0,
            "posts_saved": saved,
            "category": request.category,
            "message": "Sample data loaded successfully!"
        }
        
    except Exception as e:
        logger.error(f"Sample load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/csv/load")
async def load_csv_data(request: LoadCSVRequest):
    """
    Load and analyze data from a CSV file.
    
    Useful for Kaggle datasets. The CSV should have at least a text column.
    
    Recommended datasets:
    - Sentiment140: text column = "text"
    - Product reviews: text column = "review"
    """
    try:
        loader = get_sample_loader()
        posts = loader.load_csv(
            filepath=request.filepath,
            text_column=request.text_column,
            limit=request.limit
        )
        
        if not posts:
            raise HTTPException(status_code=400, detail="No data loaded. Check filepath and column name.")
        
        # Analyze and store
        db = get_database()
        collector = get_collector()
        ensemble = get_ensemble()
        
        analyzed = []
        for post in posts:
            text = post["selftext"] or post["title"]
            
            if request.analyze:
                sentiment = ensemble.predict(text[:1000], include_breakdown=False)
            else:
                sentiment = {"label": "neutral", "score": 0, "confidence": 0}
            
            analyzed.append({
                "post": post,
                "sentiment": sentiment
            })
            
            # Track for trends
            collector.trend_detector.add_entry(
                text=text[:200],
                sentiment=sentiment,
                source="csv_import"
            )
        
        # Save to database
        saved = db.save_posts_batch(analyzed)
        
        return {
            "success": True,
            "filepath": request.filepath,
            "posts_loaded": len(posts),
            "posts_saved": saved,
            "message": f"Loaded {len(posts)} rows from CSV"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample/quick")
async def quick_sample_load(count: int = 10):
    """Quick load sample data (GET request for easy testing)."""
    try:
        loader = get_sample_loader()
        posts = loader.get_sample_posts(count=count)
        
        db = get_database()
        ensemble = get_ensemble()
        collector = get_collector()
        
        analyzed = []
        for post in posts:
            text = post["title"] + " " + (post.get("selftext", "") or "")
            sentiment = ensemble.predict(text[:500], include_breakdown=False)
            
            analyzed.append({
                "post": post,
                "sentiment": sentiment
            })
            
            collector.trend_detector.add_entry(
                text=text[:200],
                sentiment=sentiment,
                source=post["subreddit"]
            )
        
        saved = db.save_posts_batch(analyzed)
        
        return {
            "success": True,
            "posts_loaded": count,
            "posts_saved": saved,
            "message": f"Loaded {count} sample posts for testing!"
        }
    except Exception as e:
        logger.error(f"Quick sample error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Kaggle Dataset Endpoints =====

class LoadSentiment140Request(BaseModel):
    limit: int = 100
    balanced: bool = True
    analyze: bool = True


@app.get("/api/kaggle/status")
async def kaggle_datasets_status():
    """Check which Kaggle datasets are available in the datasets folder."""
    try:
        from data.kaggle_loaders import get_available_datasets
        return get_available_datasets()
    except Exception as e:
        logger.error(f"Kaggle status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kaggle/sentiment140")
async def load_sentiment140(request: LoadSentiment140Request):
    """
    Load and analyze Sentiment140 Twitter dataset.
    
    Place the CSV file in backend/datasets/ folder first.
    Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
    """
    try:
        from data.kaggle_loaders import Sentiment140Loader
        
        loader = Sentiment140Loader()
        posts = loader.load(limit=request.limit, balanced=request.balanced)
        
        if not posts:
            raise HTTPException(
                status_code=404,
                detail="Sentiment140 dataset not found. Place CSV in backend/datasets/sentiment140.csv"
            )
        
        # Analyze and store
        db = get_database()
        collector = get_collector()
        ensemble = get_ensemble()
        
        analyzed = []
        for post in posts:
            text = post["selftext"]
            
            if request.analyze:
                sentiment = ensemble.predict(text[:500], include_breakdown=False)
            else:
                sentiment = {"label": "neutral", "score": 0, "confidence": 0}
            
            analyzed.append({
                "post": post,
                "sentiment": sentiment
            })
            
            collector.trend_detector.add_entry(
                text=text[:200],
                sentiment=sentiment,
                source="sentiment140_twitter"
            )
        
        saved = db.save_posts_batch(analyzed)
        
        return {
            "success": True,
            "dataset": "sentiment140",
            "posts_loaded": len(posts),
            "posts_saved": saved,
            "balanced": request.balanced,
            "message": f"Loaded {len(posts)} tweets from Sentiment140!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment140 load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kaggle/sentiment140/stats")
async def sentiment140_stats():
    """Get statistics about the Sentiment140 dataset."""
    try:
        from data.kaggle_loaders import Sentiment140Loader
        loader = Sentiment140Loader()
        return loader.get_stats()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)




