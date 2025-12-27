"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings, configurable via environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Path("data")

    # Logging
    log_level: LogLevel = "INFO"
    log_file_enabled: bool = False
    log_file_path: Path | None = None
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_file_backup_count: int = 5

    @property
    def resolved_log_file_path(self) -> Path:
        """Return log file path, defaulting to data_dir/vocabext.log if not set."""
        return self.log_file_path or self.data_dir / "vocabext.log"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vocab.db"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    # AnkiConnect
    anki_connect_url: str = "http://localhost:8765"
    anki_deck: str = "German::Vocabulary"
    anki_note_type: str = "German Vocabulary"

    # spaCy
    spacy_model: str = "de_core_news_lg"

    # Whisper
    whisper_model: str = "large"

    # Upload limits
    max_upload_size_mb: int = 100  # Maximum upload size in MB


settings = Settings()
