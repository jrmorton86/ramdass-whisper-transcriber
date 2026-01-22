"""
Configuration settings for the API.
"""

import sys
import os
from pathlib import Path
from pydantic_settings import BaseSettings


# Root directory (transcriber/)
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Detect pipeline Python executable (uses root venv with whisper installed)
if sys.platform == "win32":
    _PIPELINE_PYTHON = ROOT_DIR / "venv" / "Scripts" / "python.exe"
else:
    _PIPELINE_PYTHON = ROOT_DIR / "venv" / "bin" / "python"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Database
    database_url: str = "sqlite+aiosqlite:///./jobs.db"

    # File Storage
    upload_dir: Path = Path("uploads")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # Pipeline Settings
    pipeline_dir: Path = Path(__file__).parent.parent.parent / "pipeline"
    pipeline_python: Path = _PIPELINE_PYTHON  # Python with whisper installed
    gpu_ids: str = "0"
    whisper_model: str = "medium"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)
