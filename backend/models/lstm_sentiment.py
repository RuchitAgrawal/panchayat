"""
TextBlob-based Sentiment Analysis (LSTM Alternative).

Since training a custom LSTM requires labeled data and GPU resources,
we use TextBlob as a lightweight alternative that provides similar functionality.

TextBlob uses a pre-trained Naive Bayes classifier under the hood, which
serves as our "LSTM" component in the ensemble for this demo project.

For a production system, you could swap this with:
- A fine-tuned DistilBERT model
- A pre-trained LSTM from TensorFlow Hub
- VaderSentiment for social media

Reference: https://textblob.readthedocs.io/
"""
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Lazy import TextBlob
_textblob = None


def _get_textblob():
    """Lazy load TextBlob."""
    global _textblob
    if _textblob is None:
        try:
            from textblob import TextBlob
            _textblob = TextBlob
            logger.info("TextBlob loaded successfully")
        except ImportError:
            logger.warning("TextBlob not installed, using fallback")
            _textblob = None
    return _textblob


class LSTMSentiment:
    """
    TextBlob-based sentiment classifier (LSTM alternative).
    
    TextBlob provides:
    - Polarity: -1 (negative) to +1 (positive)
    - Subjectivity: 0 (objective) to 1 (subjective)
    
    We map polarity to sentiment labels:
    - polarity < -0.1: negative
    - polarity > 0.1: positive
    - otherwise: neutral
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize the sentiment classifier.
        
        Args:
            threshold: Polarity threshold for neutral classification
        """
        self.threshold = threshold
        self._textblob_class = None
    
    @property
    def textblob_class(self):
        """Lazy load TextBlob class."""
        if self._textblob_class is None:
            self._textblob_class = _get_textblob()
        return self._textblob_class
    
    def _polarity_to_sentiment(self, polarity: float, subjectivity: float) -> Dict:
        """
        Convert polarity score to sentiment label.
        
        Args:
            polarity: -1 to +1 score
            subjectivity: 0 to 1 score
            
        Returns:
            Dict with sentiment label and scores
        """
        if polarity < -self.threshold:
            label = "negative"
        elif polarity > self.threshold:
            label = "positive"
        else:
            label = "neutral"
        
        # Confidence based on how far from neutral + subjectivity
        confidence = min(abs(polarity) * subjectivity + 0.3, 1.0)
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4)
        }
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with label, confidence, and scores
        """
        if not text or not text.strip():
            return {
                "label": "neutral",
                "confidence": 0.0,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "error": "Empty text"
            }
        
        try:
            if self.textblob_class is None:
                # Fallback if TextBlob not available
                return self._fallback_predict(text)
            
            blob = self.textblob_class(text)
            return self._polarity_to_sentiment(
                blob.sentiment.polarity,
                blob.sentiment.subjectivity
            )
        except Exception as e:
            logger.error(f"TextBlob prediction error: {e}")
            return {
                "label": "neutral",
                "confidence": 0.0,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "error": str(e)
            }
    
    def _fallback_predict(self, text: str) -> Dict:
        """
        Simple keyword-based fallback if TextBlob unavailable.
        """
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'amazing', 'best', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sad', 'angry']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {"label": "positive", "confidence": 0.5, "polarity": 0.3, "subjectivity": 0.5}
        elif neg_count > pos_count:
            return {"label": "negative", "confidence": 0.5, "polarity": -0.3, "subjectivity": 0.5}
        else:
            return {"label": "neutral", "confidence": 0.5, "polarity": 0.0, "subjectivity": 0.5}
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts."""
        return [self.predict(text) for text in texts]
    
    def get_numeric_score(self, text: str) -> float:
        """
        Get a numeric sentiment score from -1 to +1.
        
        Args:
            text: Input text
            
        Returns:
            Float between -1 and 1
        """
        result = self.predict(text)
        return result.get("polarity", 0.0)


# Singleton instance
lstm_model = LSTMSentiment()


def predict_sentiment(text: str) -> Dict:
    """Convenience function for single text prediction."""
    return lstm_model.predict(text)
