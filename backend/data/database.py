"""
SQLite Database Layer for Panchayat.

Simple database for storing analyzed posts and sentiment data.
Uses SQLAlchemy for ORM and connection management.
"""
from typing import List, Dict, Optional
from datetime import datetime
import logging
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

# Database setup
DB_PATH = Path(__file__).parent.parent / "panchayat.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ===== Models =====

class Post(Base):
    """Reddit post with sentiment analysis."""
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True)  # Reddit post ID
    title = Column(String, nullable=False)
    selftext = Column(Text, default="")
    author = Column(String, default="[deleted]")
    subreddit = Column(String, nullable=False, index=True)
    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    created_utc = Column(DateTime, nullable=False)
    permalink = Column(String)
    
    # Sentiment analysis results
    sentiment_label = Column(String)  # positive/negative/neutral
    sentiment_score = Column(Float)   # -1 to +1
    sentiment_confidence = Column(Float)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "selftext": self.selftext,
            "author": self.author,
            "subreddit": self.subreddit,
            "score": self.score,
            "num_comments": self.num_comments,
            "created_utc": self.created_utc.isoformat() if self.created_utc else None,
            "permalink": self.permalink,
            "sentiment": {
                "label": self.sentiment_label,
                "score": self.sentiment_score,
                "confidence": self.sentiment_confidence
            },
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None
        }


class Comment(Base):
    """Reddit comment with sentiment analysis."""
    __tablename__ = "comments"
    
    id = Column(String, primary_key=True)  # Reddit comment ID
    post_id = Column(String, index=True)
    body = Column(Text, nullable=False)
    author = Column(String, default="[deleted]")
    score = Column(Integer, default=0)
    created_utc = Column(DateTime, nullable=False)
    
    # Sentiment
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    sentiment_confidence = Column(Float)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "post_id": self.post_id,
            "body": self.body[:200] + "..." if len(self.body) > 200 else self.body,
            "author": self.author,
            "score": self.score,
            "created_utc": self.created_utc.isoformat() if self.created_utc else None,
            "sentiment": {
                "label": self.sentiment_label,
                "score": self.sentiment_score,
                "confidence": self.sentiment_confidence
            }
        }


class CollectionJob(Base):
    """Track data collection jobs."""
    __tablename__ = "collection_jobs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    subreddit = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    posts_collected = Column(Integer, default=0)
    comments_collected = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)


# Create tables
def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized at {DB_PATH}")


# ===== Database Operations =====

class Database:
    """Database operations wrapper."""
    
    def __init__(self):
        init_db()
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return SessionLocal()
    
    # ----- Posts -----
    
    def save_post(self, post_data: Dict, sentiment: Dict) -> bool:
        """Save a post with its sentiment analysis."""
        session = self.get_session()
        try:
            post = Post(
                id=post_data["id"],
                title=post_data["title"],
                selftext=post_data.get("selftext", ""),
                author=post_data.get("author", "[deleted]"),
                subreddit=post_data["subreddit"],
                score=post_data.get("score", 0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=post_data.get("created_utc", datetime.utcnow()),
                permalink=post_data.get("permalink", ""),
                sentiment_label=sentiment.get("label"),
                sentiment_score=sentiment.get("score"),
                sentiment_confidence=sentiment.get("confidence"),
                analyzed_at=datetime.utcnow()
            )
            session.merge(post)  # Insert or update
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving post: {e}")
            return False
        finally:
            session.close()
    
    def save_posts_batch(self, posts_with_sentiment: List[Dict]) -> int:
        """Save multiple posts at once."""
        session = self.get_session()
        saved = 0
        try:
            for item in posts_with_sentiment:
                post_data = item["post"]
                sentiment = item["sentiment"]
                post = Post(
                    id=post_data["id"],
                    title=post_data["title"],
                    selftext=post_data.get("selftext", ""),
                    author=post_data.get("author", "[deleted]"),
                    subreddit=post_data["subreddit"],
                    score=post_data.get("score", 0),
                    num_comments=post_data.get("num_comments", 0),
                    created_utc=post_data.get("created_utc", datetime.utcnow()),
                    permalink=post_data.get("permalink", ""),
                    sentiment_label=sentiment.get("label"),
                    sentiment_score=sentiment.get("score"),
                    sentiment_confidence=sentiment.get("confidence"),
                    analyzed_at=datetime.utcnow()
                )
                session.merge(post)
                saved += 1
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving batch: {e}")
        finally:
            session.close()
        return saved
    
    def get_posts(self, 
                  subreddit: Optional[str] = None,
                  limit: int = 50,
                  offset: int = 0) -> List[Dict]:
        """Get posts from database."""
        session = self.get_session()
        try:
            query = session.query(Post).order_by(Post.analyzed_at.desc())
            if subreddit:
                query = query.filter(Post.subreddit == subreddit)
            posts = query.offset(offset).limit(limit).all()
            return [p.to_dict() for p in posts]
        finally:
            session.close()
    
    def get_post_count(self, subreddit: Optional[str] = None) -> int:
        """Get total post count."""
        session = self.get_session()
        try:
            query = session.query(Post)
            if subreddit:
                query = query.filter(Post.subreddit == subreddit)
            return query.count()
        finally:
            session.close()
    
    def get_sentiment_stats(self, subreddit: Optional[str] = None) -> Dict:
        """Get aggregated sentiment statistics."""
        session = self.get_session()
        try:
            query = session.query(Post)
            if subreddit:
                query = query.filter(Post.subreddit == subreddit)
            
            posts = query.all()
            if not posts:
                return {"total": 0, "avg_score": 0, "distribution": {}}
            
            scores = [p.sentiment_score for p in posts if p.sentiment_score is not None]
            labels = [p.sentiment_label for p in posts if p.sentiment_label]
            
            distribution = {}
            for label in labels:
                distribution[label] = distribution.get(label, 0) + 1
            
            return {
                "total": len(posts),
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "distribution": distribution
            }
        finally:
            session.close()
    
    def get_subreddits(self) -> List[str]:
        """Get list of all subreddits in database."""
        session = self.get_session()
        try:
            results = session.query(Post.subreddit).distinct().all()
            return [r[0] for r in results]
        finally:
            session.close()
    
    # ----- Collection Jobs -----
    
    def create_job(self, subreddit: str) -> int:
        """Create a new collection job."""
        session = self.get_session()
        try:
            job = CollectionJob(
                subreddit=subreddit,
                status="pending",
                started_at=datetime.utcnow()
            )
            session.add(job)
            session.commit()
            return job.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating job: {e}")
            return -1
        finally:
            session.close()
    
    def update_job(self, job_id: int, **kwargs):
        """Update a collection job."""
        session = self.get_session()
        try:
            job = session.query(CollectionJob).filter(CollectionJob.id == job_id).first()
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating job: {e}")
        finally:
            session.close()


# Singleton instance
db = Database()


def get_db() -> Database:
    """Get database instance."""
    return db
