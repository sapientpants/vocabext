"""Tests for application configuration."""

from pathlib import Path


class TestSettingsProperties:
    """Tests for Settings computed properties."""

    def test_resolved_log_file_path_default(self):
        """Should return default log file path when not set."""
        from app.config import Settings

        settings = Settings(data_dir=Path("/tmp/test"))
        assert settings.resolved_log_file_path == Path("/tmp/test/vocabext.log")

    def test_resolved_log_file_path_custom(self):
        """Should return custom log file path when set."""
        from app.config import Settings

        settings = Settings(
            data_dir=Path("/tmp/test"),
            log_file_path=Path("/custom/path.log"),
        )
        assert settings.resolved_log_file_path == Path("/custom/path.log")

    def test_upload_dir(self):
        """Should return uploads subdirectory."""
        from app.config import Settings

        settings = Settings(data_dir=Path("/tmp/test"))
        assert settings.upload_dir == Path("/tmp/test/uploads")

    def test_db_path(self):
        """Should return database path."""
        from app.config import Settings

        settings = Settings(data_dir=Path("/tmp/test"))
        assert settings.db_path == Path("/tmp/test/vocab.db")


class TestSettingsDefaults:
    """Tests for Settings default values."""

    def test_default_values(self, monkeypatch):
        """Should have sensible defaults when no env vars set."""
        from app.config import Settings

        # Clear environment variables that might override defaults
        monkeypatch.delenv("LOG_FILE_ENABLED", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("ANKI_DECK", raising=False)
        monkeypatch.delenv("SPACY_MODEL", raising=False)
        monkeypatch.delenv("DICTIONARY_ENABLED", raising=False)

        settings = Settings(_env_file=None)  # Ignore .env file
        assert settings.log_level == "INFO"
        assert settings.log_file_enabled is False
        assert settings.openai_model == "gpt-5-mini"
        assert settings.anki_deck == "German::Vocabulary"
        assert settings.spacy_model == "de_core_news_lg"
        assert settings.dictionary_enabled is True
