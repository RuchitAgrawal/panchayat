"""
Kaggle Dataset Loaders.

Specialized loaders for popular Kaggle sentiment datasets
with proper column mapping and preprocessing.
"""
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Default datasets directory
DATASETS_DIR = Path(__file__).parent.parent / "datasets"


class Sentiment140Loader:
    """
    Loader for Sentiment140 Twitter dataset.
    
    Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
    File: training.1600000.processed.noemoticon.csv
    
    Columns (no header):
    0: target (0=negative, 4=positive)
    1: id
    2: date
    3: flag
    4: user
    5: text
    """
    
    # Column indices (no header in original file)
    COL_TARGET = 0
    COL_ID = 1
    COL_DATE = 2
    COL_FLAG = 3
    COL_USER = 4
    COL_TEXT = 5
    
    COLUMN_NAMES = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize loader.
        
        Args:
            filepath: Path to CSV file (or will look in datasets folder)
        """
        if filepath:
            self.filepath = Path(filepath)
        else:
            # Look for common file names in datasets folder
            for name in ['sentiment140.csv', 'training.1600000.processed.noemoticon.csv']:
                path = DATASETS_DIR / name
                if path.exists():
                    self.filepath = path
                    break
            else:
                self.filepath = None
    
    def load(self, limit: int = 100, balanced: bool = True) -> List[Dict]:
        """
        Load tweets from Sentiment140 dataset.
        
        Args:
            limit: Maximum tweets to load
            balanced: If True, load equal positive/negative samples
        
        Returns:
            List of post dictionaries
        """
        if not self.filepath or not self.filepath.exists():
            logger.error(f"Sentiment140 file not found. Expected at: {DATASETS_DIR}")
            return []
        
        try:
            import pandas as pd
            
            # Load CSV (no header in original file)
            df = pd.read_csv(
                self.filepath, 
                encoding='latin-1',
                names=self.COLUMN_NAMES,
                nrows=limit * 10 if balanced else limit  # Load extra for balancing
            )
            
            # Clean data
            df = df.dropna(subset=['text'])
            df['text'] = df['text'].astype(str)
            
            # Balance if requested
            if balanced:
                pos = df[df['target'] == 4].head(limit // 2)
                neg = df[df['target'] == 0].head(limit // 2)
                df = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)
            else:
                df = df.head(limit)
            
            # Convert to post format
            results = []
            base_time = datetime.utcnow()
            
            for i, row in df.iterrows():
                sentiment_label = "positive" if row['target'] == 4 else "negative"
                
                results.append({
                    "id": f"s140_{row['id']}",
                    "title": row['text'][:100],  # First 100 chars as title
                    "selftext": row['text'],
                    "author": row['user'],
                    "subreddit": "sentiment140_twitter",
                    "score": 0,
                    "num_comments": 0,
                    "created_utc": base_time,
                    "permalink": f"/twitter/{row['id']}",
                    "original_label": sentiment_label  # Ground truth
                })
            
            logger.info(f"Loaded {len(results)} tweets from Sentiment140")
            return results
            
        except ImportError:
            logger.error("pandas not installed")
            return []
        except Exception as e:
            logger.error(f"Error loading Sentiment140: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if not self.filepath or not self.filepath.exists():
            return {"exists": False}
        
        try:
            import pandas as pd
            
            # Quick count
            df = pd.read_csv(
                self.filepath,
                encoding='latin-1', 
                names=self.COLUMN_NAMES,
                usecols=['target']
            )
            
            return {
                "exists": True,
                "total_rows": len(df),
                "positive": int((df['target'] == 4).sum()),
                "negative": int((df['target'] == 0).sum()),
                "filepath": str(self.filepath)
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}


class IMDBLoader:
    """
    Loader for IMDB Movie Reviews dataset.
    
    Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    Columns: review, sentiment
    """
    
    def __init__(self, filepath: Optional[str] = None):
        if filepath:
            self.filepath = Path(filepath)
        else:
            for name in ['imdb.csv', 'IMDB Dataset.csv']:
                path = DATASETS_DIR / name
                if path.exists():
                    self.filepath = path
                    break
            else:
                self.filepath = None
    
    def load(self, limit: int = 100) -> List[Dict]:
        """Load IMDB reviews."""
        if not self.filepath or not self.filepath.exists():
            return []
        
        try:
            import pandas as pd
            
            df = pd.read_csv(self.filepath, nrows=limit)
            
            results = []
            for i, row in df.iterrows():
                text = str(row.get('review', ''))
                sentiment = str(row.get('sentiment', '')).lower()
                
                results.append({
                    "id": f"imdb_{i}",
                    "title": text[:100],
                    "selftext": text,
                    "author": "imdb_reviewer",
                    "subreddit": "imdb_reviews",
                    "score": 0,
                    "num_comments": 0,
                    "created_utc": datetime.utcnow(),
                    "permalink": f"/imdb/{i}",
                    "original_label": sentiment
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading IMDB: {e}")
            return []


# Convenience functions
def load_sentiment140(limit: int = 100, balanced: bool = True) -> List[Dict]:
    """Load Sentiment140 dataset."""
    return Sentiment140Loader().load(limit=limit, balanced=balanced)


def load_imdb(limit: int = 100) -> List[Dict]:
    """Load IMDB dataset."""
    return IMDBLoader().load(limit=limit)


def get_available_datasets() -> Dict:
    """Check which datasets are available."""
    s140 = Sentiment140Loader()
    imdb = IMDBLoader()
    
    return {
        "sentiment140": {
            "available": s140.filepath is not None and s140.filepath.exists(),
            "path": str(s140.filepath) if s140.filepath else None
        },
        "imdb": {
            "available": imdb.filepath is not None and imdb.filepath.exists(),
            "path": str(imdb.filepath) if imdb.filepath else None
        },
        "datasets_folder": str(DATASETS_DIR)
    }
