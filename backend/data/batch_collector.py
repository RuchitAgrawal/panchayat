"""
Batch Data Collector for Reddit Posts.

Orchestrates fetching posts from Reddit, analyzing sentiment,
and storing results in the database.
"""
from typing import List, Dict, Optional
from datetime import datetime
import logging

from .reddit_client import RedditClient, reddit_client
from .database import Database, get_db

logger = logging.getLogger(__name__)


class BatchCollector:
    """
    Collects and analyzes posts from Reddit in batches.
    
    Workflow:
    1. Fetch posts from subreddit(s)
    2. Analyze sentiment for each post
    3. Store in database
    4. Track to trend detector
    """
    
    def __init__(self):
        self.reddit = reddit_client
        self.db = get_db()
        self._ensemble = None
        self._trend_detector = None
    
    @property
    def ensemble(self):
        """Lazy load sentiment ensemble."""
        if self._ensemble is None:
            from models.ensemble import SentimentEnsemble
            self._ensemble = SentimentEnsemble()
        return self._ensemble
    
    @property
    def trend_detector(self):
        """Lazy load trend detector."""
        if self._trend_detector is None:
            from nlp.trend_detector import TrendDetector
            self._trend_detector = TrendDetector()
        return self._trend_detector
    
    def collect_subreddit(self,
                          subreddit: str,
                          sort: str = "hot",
                          limit: int = 25,
                          analyze: bool = True,
                          track_trends: bool = True) -> Dict:
        """
        Collect posts from a subreddit.
        
        Args:
            subreddit: Subreddit name (without r/)
            sort: Sort method (hot, new, top, rising)
            limit: Maximum posts to fetch
            analyze: Whether to analyze sentiment
            track_trends: Whether to add to trend detector
        
        Returns:
            Collection results with stats
        """
        start_time = datetime.utcnow()
        
        # Check if Reddit is configured
        if not self.reddit.is_configured:
            return {
                "success": False,
                "error": "Reddit API not configured. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env",
                "subreddit": subreddit
            }
        
        # Create collection job
        job_id = self.db.create_job(subreddit)
        self.db.update_job(job_id, status="running")
        
        try:
            # Fetch posts
            logger.info(f"Fetching {limit} {sort} posts from r/{subreddit}")
            posts = self.reddit.fetch_posts(subreddit, sort=sort, limit=limit)
            
            if not posts:
                self.db.update_job(job_id, status="completed", posts_collected=0)
                return {
                    "success": True,
                    "subreddit": subreddit,
                    "posts_fetched": 0,
                    "posts_analyzed": 0,
                    "message": "No posts found or subreddit is private/banned"
                }
            
            # Analyze and store
            analyzed_posts = []
            for post in posts:
                # Combine title and selftext for analysis
                text = post["title"]
                if post.get("selftext"):
                    text += " " + post["selftext"][:500]  # Limit text length
                
                # Get sentiment
                if analyze:
                    sentiment = self.ensemble.predict(text, include_breakdown=False)
                else:
                    sentiment = {"label": "neutral", "score": 0, "confidence": 0}
                
                analyzed_posts.append({
                    "post": post,
                    "sentiment": sentiment
                })
                
                # Track for trends
                if track_trends and analyze:
                    self.trend_detector.add_entry(
                        text=text[:200],
                        sentiment=sentiment,
                        timestamp=post["created_utc"],
                        source=f"r/{subreddit}"
                    )
            
            # Save to database
            saved_count = self.db.save_posts_batch(analyzed_posts)
            
            # Update job
            self.db.update_job(
                job_id,
                status="completed",
                posts_collected=len(posts),
                completed_at=datetime.utcnow()
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "subreddit": subreddit,
                "sort": sort,
                "posts_fetched": len(posts),
                "posts_analyzed": len(analyzed_posts) if analyze else 0,
                "posts_saved": saved_count,
                "duration_seconds": round(duration, 2),
                "job_id": job_id
            }
            
        except Exception as e:
            logger.error(f"Collection error for r/{subreddit}: {e}")
            self.db.update_job(
                job_id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
            return {
                "success": False,
                "subreddit": subreddit,
                "error": str(e),
                "job_id": job_id
            }
    
    def collect_multiple(self,
                         subreddits: List[str],
                         sort: str = "hot",
                         limit: int = 10) -> Dict:
        """
        Collect from multiple subreddits.
        
        Args:
            subreddits: List of subreddit names
            sort: Sort method
            limit: Posts per subreddit
        
        Returns:
            Combined results
        """
        results = []
        total_posts = 0
        
        for subreddit in subreddits:
            result = self.collect_subreddit(subreddit, sort=sort, limit=limit)
            results.append(result)
            if result.get("success"):
                total_posts += result.get("posts_fetched", 0)
        
        return {
            "subreddits_processed": len(subreddits),
            "total_posts_collected": total_posts,
            "results": results
        }
    
    def get_collection_stats(self) -> Dict:
        """Get overall collection statistics."""
        subreddits = self.db.get_subreddits()
        total_posts = self.db.get_post_count()
        sentiment_stats = self.db.get_sentiment_stats()
        
        return {
            "total_posts": total_posts,
            "subreddits": subreddits,
            "subreddit_count": len(subreddits),
            "sentiment_stats": sentiment_stats,
            "trends_tracked": len(self.trend_detector.history)
        }


# Singleton instance
collector = BatchCollector()


def collect_from_subreddit(subreddit: str, limit: int = 25, sort: str = "hot") -> Dict:
    """Convenience function for collection."""
    return collector.collect_subreddit(subreddit, sort=sort, limit=limit)


def get_collection_stats() -> Dict:
    """Get collection statistics."""
    return collector.get_collection_stats()
