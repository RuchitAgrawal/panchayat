"""
Configuration module for Panchayat backend.
Loads environment variables and provides settings.
"""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Reddit API
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "panchayat:v1.0.0")
    
    # Default subreddits
    default_subreddits: list[str] = os.getenv(
        "DEFAULT_SUBREDDITS", "technology,india,news"
    ).split(",")
    
    # API settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Model settings
    bert_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    ensemble_weights: dict = {
        "bert": 0.5,
        "lstm": 0.3,
        "rf": 0.2
    }
    
    class Config:
        env_file = ".env"


settings = Settings()
