"""
N-gram Frequency Analyzer for Keyword Extraction.

Extracts frequent unigrams, bigrams, and trigrams from text collections
for topic cloud visualization and keyword detection.
"""
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


class NgramAnalyzer:
    """
    Analyzes text collections to extract frequent n-grams.
    
    Used for:
    - Topic cloud visualization (word frequency)
    - Keyword extraction for trends
    - Content summarization
    """
    
    def __init__(self, 
                 ngram_range: Tuple[int, int] = (1, 3),
                 stop_words: str = 'english',
                 max_features: int = 1000):
        """
        Initialize the analyzer.
        
        Args:
            ngram_range: Tuple of (min_n, max_n) for n-gram extraction
            stop_words: Language for stopword removal
            max_features: Maximum number of features to consider
        """
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.max_features = max_features
        
    def extract_ngrams(self, 
                       texts: List[str], 
                       top_k: int = 20,
                       ngram_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Extract top frequent n-grams from a collection of texts.
        
        Args:
            texts: List of text strings to analyze
            top_k: Number of top n-grams to return
            ngram_range: Override default n-gram range
            
        Returns:
            List of dicts with 'ngram', 'count', and 'frequency' keys
        """
        if not texts:
            return []
        
        # Filter empty texts
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []
        
        try:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range or self.ngram_range,
                stop_words=self.stop_words,
                max_features=self.max_features,
                lowercase=True
            )
            
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum counts across all documents
            counts = X.sum(axis=0).A1
            total_count = counts.sum()
            
            # Create list of (ngram, count) tuples
            ngram_counts = list(zip(feature_names, counts))
            
            # Sort by count descending
            ngram_counts.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k with frequency
            results = []
            for ngram, count in ngram_counts[:top_k]:
                results.append({
                    "ngram": ngram,
                    "count": int(count),
                    "frequency": round(count / total_count, 4) if total_count > 0 else 0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"N-gram extraction error: {e}")
            return []
    
    def extract_unigrams(self, texts: List[str], top_k: int = 20) -> List[Dict]:
        """Extract single words only."""
        return self.extract_ngrams(texts, top_k=top_k, ngram_range=(1, 1))
    
    def extract_bigrams(self, texts: List[str], top_k: int = 20) -> List[Dict]:
        """Extract two-word phrases only."""
        return self.extract_ngrams(texts, top_k=top_k, ngram_range=(2, 2))
    
    def extract_trigrams(self, texts: List[str], top_k: int = 20) -> List[Dict]:
        """Extract three-word phrases only."""
        return self.extract_ngrams(texts, top_k=top_k, ngram_range=(3, 3))
    
    def get_word_cloud_data(self, texts: List[str], max_words: int = 50) -> List[Dict]:
        """
        Get word frequency data formatted for word cloud visualization.
        
        Returns:
            List of dicts with 'text' and 'value' keys (compatible with react-wordcloud)
        """
        ngrams = self.extract_unigrams(texts, top_k=max_words)
        return [
            {"text": item["ngram"], "value": item["count"]}
            for item in ngrams
        ]


# Singleton instance
ngram_analyzer = NgramAnalyzer()


def extract_keywords(texts: List[str], top_k: int = 20) -> List[Dict]:
    """Convenience function to extract top keywords."""
    return ngram_analyzer.extract_ngrams(texts, top_k=top_k, ngram_range=(1, 2))


def get_word_cloud_data(texts: List[str], max_words: int = 50) -> List[Dict]:
    """Convenience function for word cloud data."""
    return ngram_analyzer.get_word_cloud_data(texts, max_words=max_words)
