"""
Reddit Client Wrapper using PRAW.

Provides methods for fetching posts and comments from subreddits
with error handling and rate limit awareness.
"""
from typing import List, Dict, Optional, Generator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Lazy load PRAW
_reddit = None


def get_reddit_client():
    """
    Get or create Reddit client instance.
    
    Returns None if credentials are not configured.
    """
    global _reddit
    
    if _reddit is None:
        try:
            import praw
            from config import settings
            
            if not settings.reddit_client_id or settings.reddit_client_id == "your_client_id_here":
                logger.warning("Reddit API credentials not configured")
                return None
            
            _reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent
            )
            logger.info("Reddit client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            return None
    
    return _reddit


class RedditClient:
    """
    Wrapper for Reddit API operations.
    
    Features:
    - Fetch hot/new/top posts from subreddits
    - Fetch comments from posts
    - Automatic pagination
    - Error handling
    """
    
    def __init__(self):
        """Initialize the Reddit client."""
        self.reddit = get_reddit_client()
        self._is_configured = self.reddit is not None
    
    @property
    def is_configured(self) -> bool:
        """Check if Reddit API is properly configured."""
        return self._is_configured
    
    def fetch_posts(self, 
                    subreddit_name: str,
                    sort: str = "hot",
                    limit: int = 25,
                    time_filter: str = "day") -> List[Dict]:
        """
        Fetch posts from a subreddit.
        
        Args:
            subreddit_name: Name of subreddit (without r/)
            sort: Sort method ('hot', 'new', 'top', 'rising')
            limit: Maximum posts to fetch (max 100)
            time_filter: Time filter for 'top' sort ('hour', 'day', 'week', 'month', 'year', 'all')
        
        Returns:
            List of post dictionaries
        """
        if not self._is_configured:
            return []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts based on sort method
            if sort == "hot":
                posts = subreddit.hot(limit=limit)
            elif sort == "new":
                posts = subreddit.new(limit=limit)
            elif sort == "top":
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort == "rising":
                posts = subreddit.rising(limit=limit)
            else:
                posts = subreddit.hot(limit=limit)
            
            results = []
            for post in posts:
                results.append({
                    "id": post.id,
                    "title": post.title,
                    "selftext": post.selftext or "",
                    "author": str(post.author) if post.author else "[deleted]",
                    "subreddit": subreddit_name,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.utcfromtimestamp(post.created_utc),
                    "url": post.url,
                    "permalink": f"https://reddit.com{post.permalink}",
                    "is_self": post.is_self,
                    "flair": post.link_flair_text or ""
                })
            
            logger.info(f"Fetched {len(results)} posts from r/{subreddit_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return []
    
    def fetch_comments(self, 
                       post_id: str, 
                       limit: int = 50,
                       depth: int = 1) -> List[Dict]:
        """
        Fetch comments from a post.
        
        Args:
            post_id: Reddit post ID
            limit: Maximum comments to fetch
            depth: Comment tree depth to traverse
        
        Returns:
            List of comment dictionaries
        """
        if not self._is_configured:
            return []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Skip "load more" links
            
            results = []
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body'):
                    results.append({
                        "id": comment.id,
                        "post_id": post_id,
                        "body": comment.body,
                        "author": str(comment.author) if comment.author else "[deleted]",
                        "score": comment.score,
                        "created_utc": datetime.utcfromtimestamp(comment.created_utc),
                        "is_submitter": comment.is_submitter,
                        "parent_id": comment.parent_id
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}: {e}")
            return []
    
    def fetch_subreddit_info(self, subreddit_name: str) -> Optional[Dict]:
        """Get subreddit metadata."""
        if not self._is_configured:
            return None
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            return {
                "name": subreddit.display_name,
                "title": subreddit.title,
                "description": subreddit.public_description,
                "subscribers": subreddit.subscribers,
                "created_utc": datetime.utcfromtimestamp(subreddit.created_utc),
                "over18": subreddit.over18
            }
        except Exception as e:
            logger.error(f"Error fetching subreddit info for r/{subreddit_name}: {e}")
            return None
    
    def search_posts(self, 
                     query: str, 
                     subreddit: str = "all",
                     limit: int = 25,
                     sort: str = "relevance",
                     time_filter: str = "week") -> List[Dict]:
        """
        Search for posts matching a query.
        
        Args:
            query: Search query string
            subreddit: Subreddit to search in ('all' for all)
            limit: Maximum results
            sort: Sort by 'relevance', 'hot', 'new', 'top', 'comments'
            time_filter: Time filter
        
        Returns:
            List of matching posts
        """
        if not self._is_configured:
            return []
        
        try:
            subreddit_obj = self.reddit.subreddit(subreddit)
            results = []
            
            for post in subreddit_obj.search(query, sort=sort, time_filter=time_filter, limit=limit):
                results.append({
                    "id": post.id,
                    "title": post.title,
                    "selftext": post.selftext or "",
                    "author": str(post.author) if post.author else "[deleted]",
                    "subreddit": post.subreddit.display_name,
                    "score": post.score,
                    "created_utc": datetime.utcfromtimestamp(post.created_utc),
                    "permalink": f"https://reddit.com{post.permalink}"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching posts: {e}")
            return []


# Singleton instance
reddit_client = RedditClient()


def fetch_subreddit_posts(subreddit: str, limit: int = 25, sort: str = "hot") -> List[Dict]:
    """Convenience function to fetch posts."""
    return reddit_client.fetch_posts(subreddit, sort=sort, limit=limit)


def is_reddit_configured() -> bool:
    """Check if Reddit API is configured."""
    return reddit_client.is_configured
