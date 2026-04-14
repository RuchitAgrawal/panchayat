"""
Random Forest Sentiment Classifier with TF-IDF.

This provides a traditional ML approach to complement the deep learning models.
Uses scikit-learn's RandomForestClassifier with TF-IDF vectorization.

Since we don't have labeled training data, this uses a pre-trained
approach: we bootstrap training data using TextBlob labels, then
train a Random Forest on that. For a real project, you'd use a
labeled dataset like IMDB, Sentiment140, or your own annotations.

Reference: scikit-learn text classification examples
"""
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy imports
_vectorizer = None
_classifier = None
_is_trained = False


class RFSentiment:
    """
    Random Forest sentiment classifier with TF-IDF features.
    
    This model:
    1. Converts text to TF-IDF vectors (bag of words with term frequency)
    2. Uses Random Forest for classification
    3. Outputs probability scores for each class
    
    Attributes:
        n_estimators: Number of trees in the forest
        max_features: TF-IDF max features
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_features: int = 5000,
                 model_path: Optional[str] = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.model_path = model_path
        
        self._vectorizer = None
        self._classifier = None
        self._is_trained = False
        self._labels = ["negative", "neutral", "positive"]
    
    def _init_models(self):
        """Initialize vectorizer and classifier."""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier
            
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),  # Unigrams and bigrams
                stop_words='english',
                min_df=2
            )
            
            self._classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=20,
                n_jobs=-1,
                random_state=42
            )
    
    def train(self, texts: List[str], labels: List[str]) -> Dict:
        """
        Train the model on labeled data.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (negative/neutral/positive)
            
        Returns:
            Dict with training metrics
        """
        self._init_models()
        
        logger.info(f"Training RF model on {len(texts)} samples")
        
        # Vectorize texts
        X = self._vectorizer.fit_transform(texts)
        
        # Train classifier
        self._classifier.fit(X, labels)
        self._is_trained = True
        
        # Get training accuracy
        train_acc = self._classifier.score(X, labels)
        
        logger.info(f"RF model trained. Training accuracy: {train_acc:.4f}")
        
        return {
            "samples": len(texts),
            "features": X.shape[1],
            "training_accuracy": round(train_acc, 4)
        }
    
    def bootstrap_train(self, sample_texts: Optional[List[str]] = None) -> Dict:
        """
        Bootstrap training using TextBlob labels.
        
        This is a quick way to get a working model without labeled data.
        For production, use real labeled data instead.
        
        Args:
            sample_texts: Optional list of texts to use for training
            
        Returns:
            Dict with training metrics
        """
        # Import TextBlob for labeling
        try:
            from textblob import TextBlob
        except ImportError:
            logger.error("TextBlob required for bootstrap training")
            return {"error": "TextBlob not installed"}
        
        # Default sample texts if none provided
        if sample_texts is None:
            sample_texts = self._get_default_samples()
        
        # Label using TextBlob
        labels = []
        for text in sample_texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity < -0.1:
                labels.append("negative")
            elif polarity > 0.1:
                labels.append("positive")
            else:
                labels.append("neutral")
        
        return self.train(sample_texts, labels)
    
    def _get_default_samples(self) -> List[str]:
        """
        Default training samples biased toward social-media language (Fix #18).

        The original 24 product-review sentences caused the RF to overfit on review
        vocabulary and misclassify almost all real Bluesky posts.  This expanded set
        covers politics, tech, sports, entertainment, news, and daily-life topics
        in the informal register typical of short-form social media.
        """
        return [
            # ── Positive ────────────────────────────────────────────────────
            "This is absolutely amazing and I love it!",
            "Great product, highly recommend to everyone",
            "Best experience ever, very happy with this",
            "Excellent quality and fantastic service",
            "I'm so pleased with this purchase",
            "Wonderful, exceeded all my expectations",
            "Perfect solution to my problem",
            "Outstanding performance and value",
            "just watched the new season and it was incredible omg",
            "honestly the best decision i ever made, feeling so grateful",
            "this update is fire!! the new features are exactly what i needed",
            "everyone needs to try this, it changed my life fr",
            "so proud of how far this team has come, amazing journey",
            "the concert last night was absolutely unreal, best night of my life",
            "finally got the job offer!! hard work pays off 🎉",
            "can't believe how talented this person is, true inspiration",
            "loving the new update, they really listened to user feedback",
            "the community here is so wholesome and supportive, love you all",
            "just finished the book and my mind is blown, 10/10 recommend",
            "the customer support went above and beyond, truly impressed",
            "new model just dropped and the benchmarks are insane",
            "this policy change is a huge W for everyone involved",
            "beautiful weather today, life is good",
            "huge shoutout to the dev team, you knocked it out of the park",
            "the open source community saves the day again, legend status",
            "this research paper is groundbreaking, big implications for the field",
            "proud of my city for coming together after the disaster",
            "the athlete's comeback story is genuinely moving",
            "local restaurant started composting, small wins matter",
            "new renewable energy record broken again, the future is bright",

            # ── Negative ────────────────────────────────────────────────────
            "This is terrible and I hate it",
            "Worst experience of my life, very disappointed",
            "Awful quality, complete waste of money",
            "Horrible service, never using again",
            "Very frustrating and annoying experience",
            "Complete garbage, don't buy this",
            "Extremely disappointing and useless",
            "Terrible product, broke immediately",
            "they literally lied to our faces and no one is being held accountable",
            "this update broke everything i hate when companies do this",
            "the app is completely unusable now, what happened to QA",
            "so tired of empty promises from these politicians",
            "this is such a scam, they should be ashamed",
            "the whole situation is just depressing, no hope in sight",
            "another data breach? my trust is completely gone at this point",
            "the queue has been broken for three days, zero support response",
            "spent two hours on hold just to get disconnected, furious",
            "they deleted everyone's progress with no warning or backup",
            "the misinformation spreading about this topic is genuinely dangerous",
            "cost of living is crushing people and no one seems to care",
            "laid off right before the holidays, absolutely devastating",
            "the traffic is unbearable today, already an hour late",
            "why is it always the good shows that get cancelled smh",
            "another oil spill and still no accountability, disgusting",
            "the match was fixed i swear, beyond frustrated with this league",
            "censorship is getting out of hand and it's scary",
            "watched the debate and genuinely lost faith in the whole system",
            "this team has let down the fans year after year",
            "the hospital wait times are a crisis, people are suffering",
            "prices went up again and quality went down, classic",

            # ── Neutral ─────────────────────────────────────────────────────
            "It's okay, nothing special",
            "Average product, does the job",
            "Not great but not bad either",
            "It works as expected, nothing more",
            "Standard quality, meets basic needs",
            "Typical performance, no complaints",
            "Regular product with normal features",
            "It's fine, just what I expected",
            "just got the update, will test it this weekend and report back",
            "interesting announcement, need to see how it plays out in practice",
            "the meeting went okay, some good points raised on both sides",
            "back at the office today, same as usual really",
            "article worth reading if you're into this topic",
            "weather is about average for this time of year",
            "quarterly results were in line with analyst expectations",
            "new policy takes effect next month, details in the link",
            "the patch notes are out, several bug fixes and minor changes",
            "attended the conference, got some useful contacts",
            "the game ended 1-1, pretty even contest overall",
            "the bill passed with amendments, debate continues in the senate",
            "researchers published findings on the topic, mixed conclusions",
            "the update rolled out overnight, checking compatibility now",
            "election results are still being counted in several districts",
            "the new museum wing opens to the public on Saturday",
            "the company reported revenue was flat compared to last quarter",
        ]
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with label, confidence, and probabilities
        """
        if not self._is_trained:
            # Auto-bootstrap if not trained
            logger.info("RF model not trained, bootstrapping...")
            self.bootstrap_train()
        
        if not text or not text.strip():
            return {
                "label": "neutral",
                "confidence": 0.0,
                "probabilities": {"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                "error": "Empty text"
            }
        
        try:
            # Vectorize input
            X = self._vectorizer.transform([text])
            
            # Get prediction and probabilities
            pred = self._classifier.predict(X)[0]
            probs = self._classifier.predict_proba(X)[0]
            
            # Build probability dict
            prob_dict = {}
            for i, label in enumerate(self._classifier.classes_):
                prob_dict[label] = round(float(probs[i]), 4)
            
            return {
                "label": pred,
                "confidence": round(float(max(probs)), 4),
                "probabilities": prob_dict
            }
        except Exception as e:
            logger.error(f"RF prediction error: {e}")
            return {
                "label": "neutral",
                "confidence": 0.0,
                "probabilities": {},
                "error": str(e)
            }
    
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
        probs = result.get("probabilities", {})
        
        # Weighted score: negative=-1, neutral=0, positive=+1
        score = (
            probs.get("positive", 0.33) * 1 +
            probs.get("neutral", 0.34) * 0 +
            probs.get("negative", 0.33) * -1
        )
        return round(score, 4)
    
    def save(self, path: str):
        """Save model to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self._vectorizer,
                'classifier': self._classifier,
                'is_trained': self._is_trained
            }, f)
        logger.info(f"RF model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._vectorizer = data['vectorizer']
            self._classifier = data['classifier']
            self._is_trained = data['is_trained']
        logger.info(f"RF model loaded from {path}")


# Singleton instance
rf_model = RFSentiment()


def predict_sentiment(text: str) -> Dict:
    """Convenience function for single text prediction."""
    return rf_model.predict(text)
