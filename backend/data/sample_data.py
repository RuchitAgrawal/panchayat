"""
Sample Data Loader for Testing.

Provides built-in sample data and CSV loading capabilities
for testing the sentiment analysis pipeline without Reddit API.

Recommended Kaggle datasets:
- Twitter Sentiment Analysis: https://www.kaggle.com/datasets/kazanova/sentiment140
- Reddit Comments: https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts
- Product Reviews: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Built-in sample data for immediate testing
SAMPLE_POSTS = [
    # Positive samples
    {"title": "Just finished my first machine learning project!", "text": "After months of learning, I finally built a working sentiment analyzer. The results are amazing!", "category": "technology"},
    {"title": "Python is absolutely incredible", "text": "I switched from Java to Python and my productivity has doubled. The syntax is so clean and readable.", "category": "programming"},
    {"title": "Got my dream job!", "text": "After 6 months of applications, I finally landed a software developer position. So happy!", "category": "career"},
    {"title": "This new AI tool is revolutionary", "text": "Using AI for code assistance has changed how I work. Highly recommend trying it out!", "category": "technology"},
    {"title": "Best purchase I've ever made", "text": "This mechanical keyboard is worth every penny. The typing experience is incredible.", "category": "reviews"},
    {"title": "React hooks are a game changer", "text": "Once you understand useState and useEffect, building components becomes so much easier.", "category": "programming"},
    {"title": "Loving the new update", "text": "The latest version fixed all the bugs I was experiencing. Great work by the dev team!", "category": "technology"},
    {"title": "Amazing customer support", "text": "Had an issue with my order and they resolved it within hours. Will definitely buy again.", "category": "reviews"},
    
    # Negative samples
    {"title": "Frustrated with this framework", "text": "Spent 3 hours debugging a simple issue because the documentation is terrible.", "category": "programming"},
    {"title": "Worst experience ever", "text": "The product broke after one week and customer service won't respond to my emails.", "category": "reviews"},
    {"title": "This update ruined everything", "text": "Why do companies push updates that break perfectly working features? So annoying.", "category": "technology"},
    {"title": "Job hunting is exhausting", "text": "100+ applications and only 2 responses. The market is brutal right now.", "category": "career"},
    {"title": "Hate this new design", "text": "The UI redesign is awful. Everything is harder to find now. Please bring back the old version.", "category": "technology"},
    {"title": "Complete waste of money", "text": "The product looks nothing like the photos. Returning immediately.", "category": "reviews"},
    {"title": "Memory leaks everywhere", "text": "This library has so many memory leaks it's unusable in production.", "category": "programming"},
    {"title": "Support ticket ignored for weeks", "text": "Still waiting for a response about my billing issue. Worst customer service I've experienced.", "category": "reviews"},
    
    # Neutral samples
    {"title": "Question about Python decorators", "text": "Can someone explain how @property works? I've read the docs but still confused.", "category": "programming"},
    {"title": "Looking for laptop recommendations", "text": "Need a laptop for web development. Budget is around $1000. Any suggestions?", "category": "technology"},
    {"title": "Comparing cloud providers", "text": "AWS vs GCP vs Azure - what are the main differences for small projects?", "category": "technology"},
    {"title": "Interview next week", "text": "Have a technical interview scheduled. Any tips for system design questions?", "category": "career"},
    {"title": "New JavaScript features", "text": "ES2024 is introducing some new array methods. Here's a summary of what's coming.", "category": "programming"},
    {"title": "Database migration question", "text": "Moving from MySQL to PostgreSQL. What should I watch out for?", "category": "programming"},
]


class SampleDataLoader:
    """
    Load sample data for testing sentiment analysis.
    
    Supports:
    - Built-in sample data (no setup required)
    - CSV file loading (for Kaggle datasets)
    """
    
    def __init__(self):
        self.samples = SAMPLE_POSTS.copy()
    
    def get_sample_posts(self, 
                         count: int = 20,
                         category: Optional[str] = None,
                         shuffle: bool = True) -> List[Dict]:
        """
        Get sample posts for testing.
        
        Args:
            count: Number of posts to return
            category: Filter by category (programming, technology, reviews, career)
            shuffle: Randomize order
        
        Returns:
            List of post dictionaries
        """
        posts = self.samples.copy()
        
        # Filter by category
        if category:
            posts = [p for p in posts if p.get("category") == category]
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(posts)
        
        # Generate full post structure
        result = []
        base_time = datetime.utcnow()
        
        for i, post in enumerate(posts[:count]):
            result.append({
                "id": f"sample_{i}_{random.randint(1000, 9999)}",
                "title": post["title"],
                "selftext": post["text"],
                "author": f"sample_user_{random.randint(1, 100)}",
                "subreddit": f"sample_{post['category']}",
                "score": random.randint(1, 500),
                "num_comments": random.randint(0, 50),
                "created_utc": base_time - timedelta(hours=random.randint(1, 72)),
                "permalink": f"/r/sample/{post['category']}/comments/abc123",
            })
        
        return result
    
    def load_csv(self, 
                 filepath: str,
                 text_column: str = "text",
                 title_column: Optional[str] = None,
                 limit: int = 100) -> List[Dict]:
        """
        Load posts from a CSV file.
        
        Args:
            filepath: Path to CSV file
            text_column: Column name containing the text
            title_column: Column name for title (optional)
            limit: Maximum rows to load
        
        Returns:
            List of post dictionaries
        """
        try:
            import pandas as pd
            
            df = pd.read_csv(filepath, nrows=limit)
            
            if text_column not in df.columns:
                logger.error(f"Column '{text_column}' not found in CSV")
                return []
            
            result = []
            base_time = datetime.utcnow()
            
            for i, row in df.iterrows():
                text = str(row[text_column])
                title = str(row[title_column]) if title_column and title_column in df.columns else text[:100]
                
                result.append({
                    "id": f"csv_{i}",
                    "title": title[:200],
                    "selftext": text,
                    "author": "csv_import",
                    "subreddit": "csv_dataset",
                    "score": 0,
                    "num_comments": 0,
                    "created_utc": base_time - timedelta(minutes=i),
                    "permalink": f"/csv/{i}",
                })
            
            logger.info(f"Loaded {len(result)} rows from CSV")
            return result
            
        except ImportError:
            logger.error("pandas not installed. Run: pip install pandas")
            return []
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return []
    
    def get_categories(self) -> List[str]:
        """Get available sample categories."""
        return list(set(p["category"] for p in self.samples))


# Singleton instance
sample_loader = SampleDataLoader()


def get_sample_posts(count: int = 20, category: Optional[str] = None) -> List[Dict]:
    """Convenience function for getting sample posts."""
    return sample_loader.get_sample_posts(count=count, category=category)


def load_csv_data(filepath: str, text_column: str = "text", limit: int = 100) -> List[Dict]:
    """Convenience function for loading CSV data."""
    return sample_loader.load_csv(filepath, text_column=text_column, limit=limit)
