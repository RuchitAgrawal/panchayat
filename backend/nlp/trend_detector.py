"""
Trend Detection and Time-Series Aggregation.

Tracks sentiment over time and aggregates data for trend visualization.
Provides rolling averages and time-bucketed sentiment scores.
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TrendDetector:
    """
    Tracks and aggregates sentiment data over time.
    
    Features:
    - Add sentiment entries with timestamps
    - Aggregate by time periods (hour, day, week)
    - Calculate rolling averages
    - Detect trend direction (rising/falling/stable)
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize the trend detector.
        
        Args:
            max_history: Maximum entries to keep in memory
        """
        self.max_history = max_history
        self.history: List[Dict] = []
        
    def add_entry(self, 
                  text: str, 
                  sentiment: Dict, 
                  timestamp: Optional[datetime] = None,
                  source: Optional[str] = None) -> None:
        """
        Add a sentiment entry to history.
        
        Args:
            text: Original text (truncated for storage)
            sentiment: Sentiment prediction dict with 'label', 'score', 'confidence'
            timestamp: Entry timestamp (defaults to now)
            source: Optional source identifier (e.g., subreddit name)
        """
        entry = {
            "timestamp": timestamp or datetime.utcnow(),
            "text": text[:200] if len(text) > 200 else text,
            "label": sentiment.get("label", "neutral"),
            "score": sentiment.get("score", 0.0),
            "confidence": sentiment.get("confidence", 0.0),
            "source": source
        }
        
        self.history.append(entry)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def add_batch(self, 
                  entries: List[Dict], 
                  source: Optional[str] = None) -> int:
        """
        Add multiple entries at once.
        
        Args:
            entries: List of dicts with 'text', 'sentiment', optional 'timestamp'
            source: Optional source identifier
            
        Returns:
            Number of entries added
        """
        count = 0
        for entry in entries:
            self.add_entry(
                text=entry.get("text", ""),
                sentiment=entry.get("sentiment", {}),
                timestamp=entry.get("timestamp"),
                source=source
            )
            count += 1
        return count
    
    def get_trends(self, 
                   period: str = "1h",
                   limit: int = 24) -> List[Dict]:
        """
        Get aggregated sentiment trends over time.
        
        Args:
            period: Aggregation period ('1h', '6h', '1d', '1w')
            limit: Maximum number of time buckets to return
            
        Returns:
            List of dicts with 'time', 'avg_score', 'count', 'label_distribution'
        """
        if not self.history:
            return []
        
        # Parse period
        period_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1)
        }
        bucket_size = period_map.get(period, timedelta(hours=1))
        
        # Group entries by time bucket
        buckets = defaultdict(list)
        for entry in self.history:
            ts = entry["timestamp"]
            # Round down to bucket
            bucket_ts = ts.replace(
                minute=0 if bucket_size >= timedelta(hours=1) else ts.minute,
                second=0, 
                microsecond=0
            )
            if bucket_size >= timedelta(days=1):
                bucket_ts = bucket_ts.replace(hour=0)
            
            buckets[bucket_ts].append(entry)
        
        # Aggregate each bucket
        trends = []
        for bucket_ts in sorted(buckets.keys())[-limit:]:
            entries = buckets[bucket_ts]
            
            # Calculate averages
            scores = [e["score"] for e in entries]
            confidences = [e["confidence"] for e in entries]
            
            # Count labels
            label_counts = defaultdict(int)
            for e in entries:
                label_counts[e["label"]] += 1
            
            trends.append({
                "time": bucket_ts.isoformat(),
                "avg_score": round(sum(scores) / len(scores), 4),
                "avg_confidence": round(sum(confidences) / len(confidences), 4),
                "count": len(entries),
                "label_distribution": dict(label_counts)
            })
        
        return trends
    
    def get_summary(self) -> Dict:
        """
        Get overall summary statistics.
        
        Returns:
            Dict with total_count, avg_score, label_percentages, trend_direction
        """
        if not self.history:
            return {
                "total_count": 0,
                "avg_score": 0,
                "label_percentages": {},
                "trend_direction": "stable"
            }
        
        scores = [e["score"] for e in self.history]
        
        # Label counts
        label_counts = defaultdict(int)
        for e in self.history:
            label_counts[e["label"]] += 1
        
        total = len(self.history)
        label_percentages = {
            label: round(count / total * 100, 1)
            for label, count in label_counts.items()
        }
        
        # Determine trend direction (compare first half vs second half)
        half = len(scores) // 2
        if half > 0:
            first_half_avg = sum(scores[:half]) / half
            second_half_avg = sum(scores[half:]) / (len(scores) - half)
            diff = second_half_avg - first_half_avg
            
            if diff > 0.1:
                trend_direction = "rising"
            elif diff < -0.1:
                trend_direction = "falling"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
        
        return {
            "total_count": total,
            "avg_score": round(sum(scores) / total, 4),
            "label_percentages": label_percentages,
            "trend_direction": trend_direction
        }
    
    def get_recent(self, limit: int = 20) -> List[Dict]:
        """
        Get most recent entries.
        
        Args:
            limit: Number of entries to return
            
        Returns:
            List of recent sentiment entries
        """
        return [
            {
                "time": e["timestamp"].isoformat(),
                "text": e["text"],
                "label": e["label"],
                "score": e["score"],
                "source": e["source"]
            }
            for e in reversed(self.history[-limit:])
        ]
    
    def clear(self) -> None:
        """Clear all history."""
        self.history = []


# Singleton instance
trend_detector = TrendDetector()


def add_sentiment_entry(text: str, sentiment: Dict, 
                        timestamp: Optional[datetime] = None,
                        source: Optional[str] = None) -> None:
    """Convenience function to add entry."""
    trend_detector.add_entry(text, sentiment, timestamp, source)


def get_trends(period: str = "1h", limit: int = 24) -> List[Dict]:
    """Convenience function to get trends."""
    return trend_detector.get_trends(period, limit)


def get_summary() -> Dict:
    """Convenience function to get summary."""
    return trend_detector.get_summary()
