"""
Configuration settings for LEGO Assembly Refactored.
Manages environment variables and application configuration.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # VLM Configuration (single model, no fallbacks)
    gemini_api_key: Optional[str] = None
    vlm_model: str = "gemini/gemini-robotics-er-1.5-preview"
    vlm_max_retries: int = 3
    vlm_timeout: int = 60  # seconds

    # Paths (relative to project root)
    data_dir: Path = Path("./data")
    manual_dir: Path = Path("./data/manuals")
    processed_dir: Path = Path("./data/processed")
    cropped_dir: Path = Path("./data/cropped")
    brick_library_dir: Path = Path("./data/brick_library")

    # Backend Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_url: str = "http://localhost:3000"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields in .env for future use


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
