"""Application configuration using Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, configurable via environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Path("data")

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vocab.db"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "ministral-3:14b"

    # AnkiConnect
    anki_connect_url: str = "http://localhost:8765"
    anki_deck: str = "German::Work Vocabulary"
    anki_note_type: str = "German Vocabulary"

    # spaCy
    spacy_model: str = "de_core_news_lg"

    # Whisper
    whisper_model: str = "large"


settings = Settings()
