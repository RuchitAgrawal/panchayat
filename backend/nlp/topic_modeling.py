"""
Topic Modeling using BERTopic.

Extracts semantic topics from text collections using transformer embeddings
and clustering. Provides topic keywords and document-topic mapping.
"""
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Lazy load BERTopic (heavy import)
_model = None


def get_topic_model():
    """Lazy load BERTopic model."""
    global _model
    if _model is None:
        from bertopic import BERTopic
        logger.info("Initializing BERTopic model...")
        _model = BERTopic(
            language="english",
            min_topic_size=3,  # Minimum documents per topic
            nr_topics="auto",  # Auto-determine number of topics
            verbose=False
        )
        logger.info("BERTopic model initialized")
    return _model


class TopicModeler:
    """
    BERTopic wrapper for semantic topic extraction.
    
    Uses BERT embeddings to cluster documents into topics,
    then extracts representative keywords for each topic.
    """
    
    def __init__(self, min_topic_size: int = 3):
        """
        Initialize the topic modeler.
        
        Args:
            min_topic_size: Minimum documents required to form a topic
        """
        self.min_topic_size = min_topic_size
        self._model = None
        self._is_fitted = False
        
    @property
    def model(self):
        """Lazy load BERTopic model."""
        if self._model is None:
            from bertopic import BERTopic
            self._model = BERTopic(
                language="english",
                min_topic_size=self.min_topic_size,
                nr_topics="auto",
                verbose=False
            )
        return self._model
    
    def extract_topics(self, 
                       texts: List[str], 
                       top_n_words: int = 10) -> Dict:
        """
        Extract topics from a collection of texts.
        
        Args:
            texts: List of documents to analyze
            top_n_words: Number of words per topic to return
            
        Returns:
            Dict with topics, topic_words, and document_topics
        """
        if not texts or len(texts) < self.min_topic_size:
            return {
                "topics": [],
                "topic_words": {},
                "document_topics": [],
                "error": f"Need at least {self.min_topic_size} documents"
            }
        
        # Filter empty texts
        valid_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if len(valid_texts) < self.min_topic_size:
            return {
                "topics": [],
                "topic_words": {},
                "document_topics": [],
                "error": f"Only {len(valid_texts)} valid documents, need {self.min_topic_size}"
            }
        
        indices, filtered_texts = zip(*valid_texts)
        
        try:
            # Fit and transform
            topics, probs = self.model.fit_transform(list(filtered_texts))
            self._is_fitted = True
            
            # Get topic info
            topic_info = self.model.get_topic_info()
            
            # Build topic_words dict
            topic_words = {}
            for topic_id in set(topics):
                if topic_id != -1:  # Skip outlier topic
                    words = self.model.get_topic(topic_id)
                    if words:
                        topic_words[str(topic_id)] = [
                            {"word": w, "score": round(s, 4)} 
                            for w, s in words[:top_n_words]
                        ]
            
            # Build topics list for visualization
            topics_list = []
            for _, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    topic_data = {
                        "id": int(row['Topic']),
                        "count": int(row['Count']),
                        "name": row.get('Name', f"Topic {row['Topic']}"),
                    }
                    # Add representative words
                    if str(row['Topic']) in topic_words:
                        topic_data["keywords"] = [
                            w["word"] for w in topic_words[str(row['Topic'])][:5]
                        ]
                    topics_list.append(topic_data)
            
            # Map document to topic (restore original indices)
            doc_topics = [-1] * len(texts)
            for orig_idx, topic_id in zip(indices, topics):
                doc_topics[orig_idx] = int(topic_id)
            
            return {
                "topics": topics_list,
                "topic_words": topic_words,
                "document_topics": doc_topics,
                "outliers": doc_topics.count(-1)
            }
            
        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return {
                "topics": [],
                "topic_words": {},
                "document_topics": [],
                "error": str(e)
            }
    
    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[Dict]:
        """
        Get keywords for a specific topic.
        
        Args:
            topic_id: Topic ID to get keywords for
            top_n: Number of keywords to return
            
        Returns:
            List of dicts with 'word' and 'score' keys
        """
        if not self._is_fitted:
            return []
        
        try:
            words = self.model.get_topic(topic_id)
            if words:
                return [
                    {"word": w, "score": round(s, 4)} 
                    for w, s in words[:top_n]
                ]
        except Exception as e:
            logger.error(f"Error getting topic keywords: {e}")
        
        return []


# Singleton instance
topic_modeler = TopicModeler()


def extract_topics(texts: List[str], top_n_words: int = 10) -> Dict:
    """Convenience function for topic extraction."""
    return topic_modeler.extract_topics(texts, top_n_words=top_n_words)
