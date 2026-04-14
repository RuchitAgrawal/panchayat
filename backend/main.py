import sys, os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.insert(0, os.path.dirname(__file__))
from db import get_db_connection   # Fix #13 — single source of truth for DB access

app = FastAPI(title="Panchayat Data Lake API")

# ── Rate Limiter (Fix #7) ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — driven by environment variable (Fix #6) ────────────────────────────
# Set ALLOWED_ORIGINS="https://yourdomain.com" in production
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
)
origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models (Fix #9) ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      description="Text to analyse (1–5000 characters)")


# ── DB Helper ─────────────────────────────────────────────────────────────────
def query_db(query: str, args=(), one: bool = False):
    """Execute a SELECT and return plain dicts — uses the shared WAL connection."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, args)
        rows = [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"DB Error: {e}")
        rows = []
    finally:
        conn.close()
    return (rows[0] if rows else None) if one else rows


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
@limiter.limit("10/minute")                          # Fix #7 – rate-limit heavy endpoint
async def analyze(req: AnalyzeRequest, request: Request):
    """
    Run the full BERT+LSTM+RF ensemble on user-submitted text.
    Offloaded to a thread pool so the async event loop is never blocked (Fix #8).
    """
    from models.ensemble import analyze_sentiment
    result = await run_in_threadpool(analyze_sentiment, req.text)   # Fix #8
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/posts")
async def get_posts(limit: int = 20):
    posts = query_db(
        "SELECT * FROM posts_recent ORDER BY timestamp DESC LIMIT ?", [limit]
    )
    formatted_posts = [
        {
            "id": p["id"],
            "title": p["text"],     # frontend expects 'title'
            "subreddit": p["source"],
            "sentiment": {"label": p["label"], "score": p["score"]},
        }
        for p in posts
    ]
    return {"posts": formatted_posts}


@app.get("/api/posts/stats")
async def get_stats():
    stats = query_db(
        "SELECT * FROM sentiment_stats ORDER BY id DESC LIMIT 1", one=True
    )
    if not stats:
        return {
            "stats": {
                "total": 0,
                "distribution": {"positive": 0, "negative": 0, "neutral": 0},
            }
        }

    total = stats["total_count"]
    return {
        "stats": {
            "total": total,
            "distribution": {
                "positive": int((stats["positive_pct"] / 100) * total),
                "negative": int((stats["negative_pct"] / 100) * total),
                "neutral":  int((stats["neutral_pct"]  / 100) * total),
            },
        }
    }


@app.get("/api/trends")
async def get_trends(period: str = "1h"):
    """
    Return time-bucketed sentiment trends filtered by period (Fix #16).

    period values: 1h | 6h | 1d | 1w
    Times in the DB are stored in IST (UTC+5:30) as 'YYYY-MM-DD HH:MM'.
    """
    period_hours = {"1h": 1, "6h": 6, "1d": 24, "1w": 168}
    hours = period_hours.get(period, 1)

    # strftime keeps the same 'YYYY-MM-DD HH:MM' format as stored values.
    # '+5 hours +30 minutes' converts 'now' (UTC) to IST, then we subtract the period.
    trends = query_db(
        """
        SELECT * FROM hourly_trends
        WHERE time >= strftime('%Y-%m-%d %H:%M', 'now',
                               '+5 hours', '+30 minutes',
                               :offset)
        ORDER BY time ASC
        """,
        {"offset": f"-{hours} hours"},
    )

    direction = "stable"
    if len(trends) >= 2:
        diff = trends[-1]["avg_score"] - trends[-2]["avg_score"]
        if diff > 0.05:
            direction = "rising"
        elif diff < -0.05:
            direction = "falling"

    recent_avg = trends[-1]["avg_score"] if trends else 0.0

    return {
        "trends": trends,
        "summary": {"trend_direction": direction, "avg_score": recent_avg},
    }


@app.get("/api/topics")
async def get_topics(limit: int = 20):
    topics = query_db(
        "SELECT * FROM trending_topics ORDER BY count DESC LIMIT ?", [limit]
    )
    return {"topics": topics}


@app.get("/api/sample/quick")
async def load_quick_sample(count: int = 10):
    """Fallback endpoint retained for compatibility — live pipeline is preferred."""
    return {"status": "success", "message": "Using live spark pipelines instead."}
