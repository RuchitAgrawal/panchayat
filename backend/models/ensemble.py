"""
Ensemble Sentiment Classifier.

Combines BERT, LSTM (TextBlob), and Random Forest predictions
using weighted voting for more robust sentiment classification.

Weights:
- BERT: 0.5 (most accurate)
- LSTM/TextBlob: 0.3 (good for short text)
- Random Forest: 0.2 (traditional ML baseline)
"""
from typing import Dict, List, Optional
import logging

from .bert_sentiment import BertSentiment
from .lstm_sentiment import LSTMSentiment
from .rf_sentiment import RFSentiment

logger = logging.getLogger(__name__)


class SentimentEnsemble:
    """
    Ensemble classifier combining BERT, LSTM, and Random Forest.
    
    Uses weighted voting to combine predictions from all three models.
    Falls back gracefully if any model fails.
    
    Attributes:
        weights: Dict of model weights (should sum to 1.0)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble.
        
        Args:
            weights: Optional custom weights for each model
        """
        self.weights = weights or {
            "bert": 0.5,
            "lstm": 0.3,
            "rf": 0.2
        }
        
        # Initialize models (lazy loading)
        self._bert = None
        self._lstm = None
        self._rf = None
        self._initialized = False
    
    @property
    def bert(self):
        if self._bert is None:
            self._bert = BertSentiment()
        return self._bert
    
    @property
    def lstm(self):
        if self._lstm is None:
            self._lstm = LSTMSentiment()
        return self._lstm
    
    @property
    def rf(self):
        if self._rf is None:
            self._rf = RFSentiment()
        return self._rf
    
    def _label_to_score(self, label: str) -> float:
        """Convert label to numeric score."""
        mapping = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        return mapping.get(label, 0.0)
    
    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to label."""
        if score < -0.33:
            return "negative"
        elif score > 0.33:
            return "positive"
        else:
            return "neutral"
    
    def predict(self, text: str, include_breakdown: bool = True) -> Dict:
        """
        Predict sentiment using ensemble voting.
        
        Args:
            text: Input text
            include_breakdown: Include individual model results
            
        Returns:
            Dict with ensemble prediction, confidence, and optional breakdown
        """
        if not text or not text.strip():
            return {
                "label": "neutral",
                "confidence": 0.0,
                "score": 0.0,
                "error": "Empty text"
            }
        
        results = {}
        scores = {}
        confidences = {}
        errors = []
        
        # Get BERT prediction
        try:
            bert_result = self.bert.predict(text)
            results["bert"] = bert_result
            scores["bert"] = self._label_to_score(bert_result.get("label", "neutral"))
            confidences["bert"] = bert_result.get("confidence", 0.0)
        except Exception as e:
            logger.error(f"BERT error: {e}")
            errors.append(f"bert: {str(e)}")
            scores["bert"] = 0.0
            confidences["bert"] = 0.0
        
        # Get LSTM/TextBlob prediction
        try:
            lstm_result = self.lstm.predict(text)
            results["lstm"] = lstm_result
            scores["lstm"] = self._label_to_score(lstm_result.get("label", "neutral"))
            confidences["lstm"] = lstm_result.get("confidence", 0.0)
        except Exception as e:
            logger.error(f"LSTM error: {e}")
            errors.append(f"lstm: {str(e)}")
            scores["lstm"] = 0.0
            confidences["lstm"] = 0.0
        
        # Get RF prediction
        try:
            rf_result = self.rf.predict(text)
            results["rf"] = rf_result
            scores["rf"] = self._label_to_score(rf_result.get("label", "neutral"))
            confidences["rf"] = rf_result.get("confidence", 0.0)
        except Exception as e:
            logger.error(f"RF error: {e}")
            errors.append(f"rf: {str(e)}")
            scores["rf"] = 0.0
            confidences["rf"] = 0.0
        
        # Calculate weighted ensemble score
        total_weight = sum(self.weights.values())
        ensemble_score = 0.0
        ensemble_confidence = 0.0
        
        for model, weight in self.weights.items():
            ensemble_score += scores.get(model, 0.0) * weight
            ensemble_confidence += confidences.get(model, 0.0) * weight
        
        ensemble_score /= total_weight
        ensemble_confidence /= total_weight
        
        # Build response
        response = {
            "label": self._score_to_label(ensemble_score),
            "confidence": round(ensemble_confidence, 4),
            "score": round(ensemble_score, 4),  # -1 to +1
        }
        
        if include_breakdown:
            response["breakdown"] = {
                "bert": {
                    "label": results.get("bert", {}).get("label", "error"),
                    "confidence": round(confidences.get("bert", 0.0), 4),
                    "weight": self.weights["bert"]
                },
                "lstm": {
                    "label": results.get("lstm", {}).get("label", "error"),
                    "confidence": round(confidences.get("lstm", 0.0), 4),
                    "weight": self.weights["lstm"]
                },
                "rf": {
                    "label": results.get("rf", {}).get("label", "error"),
                    "confidence": round(confidences.get("rf", 0.0), 4),
                    "weight": self.weights["rf"]
                }
            }
        
        if errors:
            response["warnings"] = errors
        
        return response
    
    def predict_batch(self, texts: List[str], include_breakdown: bool = False) -> List[Dict]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            include_breakdown: Include individual model results
            
        Returns:
            List of prediction dicts
        """
        return [self.predict(text, include_breakdown) for text in texts]
    
    def analyze(self, text: str) -> Dict:
        """
        Full analysis with preprocessing and prediction.
        
        Args:
            text: Raw input text
            
        Returns:
            Dict with cleaned text and prediction
        """
        from nlp.preprocessor import clean_text
        
        cleaned = clean_text(text)
        prediction = self.predict(cleaned, include_breakdown=True)
        
        return {
            "original_text": text[:200] + "..." if len(text) > 200 else text,
            "cleaned_text": cleaned[:200] + "..." if len(cleaned) > 200 else cleaned,
            **prediction
        }


# Singleton instance
ensemble = SentimentEnsemble()


def analyze_sentiment(text: str) -> Dict:
    """Convenience function for full sentiment analysis."""
    return ensemble.analyze(text)


def predict_sentiment(text: str) -> Dict:
    """Convenience function for prediction only."""
    return ensemble.predict(text)
