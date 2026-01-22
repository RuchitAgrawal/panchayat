"""
Text Preprocessing Utilities for Sentiment Analysis.

This module provides text cleaning and normalization functions.
Reference: Common preprocessing steps from NLP pipelines.
"""
import re
import string
from typing import List


class TextPreprocessor:
    """
    Preprocesses text for sentiment analysis models.
    
    Handles:
    - URL removal
    - Mention removal (@user)
    - Hashtag cleaning
    - Emoji handling
    - Lowercasing
    - Punctuation removal (optional)
    - Extra whitespace removal
    """
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = False,  # Keep hashtag text, remove #
                 lowercase: bool = True,
                 remove_punctuation: bool = False,  # Keep for BERT
                 remove_numbers: bool = False):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.whitespace_pattern = re.compile(r'\s+')
        self.number_pattern = re.compile(r'\d+')
    
    def clean(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Handle hashtags - keep the word, remove the #
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            text = self.hashtag_pattern.sub(r'\1', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        text = text.strip()
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]


# Singleton instance for easy import
preprocessor = TextPreprocessor()


def clean_text(text: str) -> str:
    """Convenience function for cleaning a single text."""
    return preprocessor.clean(text)


def clean_texts(texts: List[str]) -> List[str]:
    """Convenience function for cleaning multiple texts."""
    return preprocessor.clean_batch(texts)
