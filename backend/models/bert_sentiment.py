"""
BERT-based Sentiment Analysis Model.

Uses HuggingFace Transformers pre-trained model.
Model: nlptown/bert-base-multilingual-uncased-sentiment (1-5 stars)

Reference: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
"""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Lazy loading to avoid slow startup
_pipeline = None
_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"


def _get_pipeline():
    """Lazy load the transformers pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            logger.info(f"Loading BERT model: {_model_name}")
            _pipeline = pipeline(
                "sentiment-analysis",
                model=_model_name,
                tokenizer=_model_name
            )
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    return _pipeline


class BertSentiment:
    """
    BERT-based sentiment classifier.
    
    This model outputs 1-5 star ratings which we convert to:
    - 1-2 stars: negative
    - 3 stars: neutral  
    - 4-5 stars: positive
    
    Attributes:
        model_name: HuggingFace model identifier
        max_length: Maximum token length (BERT limit is 512)
    """
    
    def __init__(self, model_name: Optional[str] = None, max_length: int = 512):
        self.model_name = model_name or _model_name
        self.max_length = max_length
        self._classifier = None
    
    @property
    def classifier(self):
        """Lazy load classifier on first use."""
        if self._classifier is None:
            self._classifier = _get_pipeline()
        return self._classifier
    
    def _star_to_sentiment(self, label: str, score: float) -> Dict:
        """
        Convert star rating to sentiment label.
        
        Args:
            label: Star rating label (e.g., "5 stars")
            score: Confidence score
            
        Returns:
            Dict with sentiment label, confidence, and raw score
        """
        # Extract star number from label like "5 stars"
        stars = int(label.split()[0])
        
        # Map stars to sentiment
        if stars <= 2:
            sentiment = "negative"
        elif stars == 3:
            sentiment = "neutral"
        else:
            sentiment = "positive"
        
        # Normalize confidence based on how far from neutral
        # Stars 1,5 = high confidence, 2,3,4 = lower confidence
        confidence_boost = abs(stars - 3) / 2  # 0 to 1
        adjusted_confidence = score * (0.7 + 0.3 * confidence_boost)
        
        return {
            "label": sentiment,
            "confidence": round(adjusted_confidence, 4),
            "raw_stars": stars,
            "raw_score": round(score, 4)
        }
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text (will be truncated to max_length)
            
        Returns:
            Dict with label, confidence, and raw scores
        """
        if not text or not text.strip():
            return {
                "label": "neutral",
                "confidence": 0.0,
                "raw_stars": 3,
                "raw_score": 0.0,
                "error": "Empty text"
            }
        
        try:
            # Truncate text to max length (rough estimate: 4 chars per token)
            truncated = text[:self.max_length * 4]
            result = self.classifier(truncated)[0]
            return self._star_to_sentiment(result["label"], result["score"])
        except Exception as e:
            logger.error(f"BERT prediction error: {e}")
            return {
                "label": "neutral",
                "confidence": 0.0,
                "raw_stars": 3,
                "raw_score": 0.0,
                "error": str(e)
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dicts
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def get_numeric_score(self, text: str) -> float:
        """
        Get a numeric sentiment score from -1 (negative) to +1 (positive).
        
        Args:
            text: Input text
            
        Returns:
            Float between -1 and 1
        """
        result = self.predict(text)
        stars = result.get("raw_stars", 3)
        # Convert 1-5 stars to -1 to +1
        return (stars - 3) / 2


# Singleton instance
bert_model = BertSentiment()


def predict_sentiment(text: str) -> Dict:
    """Convenience function for single text prediction."""
    return bert_model.predict(text)
