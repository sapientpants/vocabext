"""Tests for service modules."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enricher import Enricher, EnrichmentResult
from app.services.extractor import TextExtractor
from app.services.llm import get_semaphore


class TestEnrichmentResult:
    """Tests for EnrichmentResult Pydantic model."""

    def test_default_values(self):
        """Should have correct default values."""
        result = EnrichmentResult()
        assert result.gender is None
        assert result.plural is None
        assert result.preterite is None
        assert result.past_participle is None
        assert result.auxiliary is None
        assert result.translations == []

    def test_with_values(self):
        """Should store provided values."""
        result = EnrichmentResult(
            gender="die",
            plural="Arbeiten",
            translations=["work", "job"],
        )
        assert result.gender == "die"
        assert result.plural == "Arbeiten"
        assert result.translations == ["work", "job"]


class TestEnricher:
    """Tests for Enricher service."""

    def test_init_defaults(self):
        """Should use default settings from config."""
        enricher = Enricher()
        assert enricher.model is None  # Uses central llm module default
        assert enricher.api_key is None

    def test_init_custom(self):
        """Should accept custom settings."""
        enricher = Enricher(api_key="test-key", model="custom-model")
        assert enricher.api_key == "test-key"
        assert enricher.model == "custom-model"

    def test_build_prompt_noun(self):
        """Should build correct prompt for nouns."""
        enricher = Enricher()
        prompt = enricher._build_prompt("Arbeit", "NOUN", "Die Arbeit ist wichtig.")
        assert "Noun" in prompt
        assert "Arbeit" in prompt
        assert "Gender" in prompt
        assert "Plural" in prompt

    def test_build_prompt_verb(self):
        """Should build correct prompt for verbs."""
        enricher = Enricher()
        prompt = enricher._build_prompt("arbeiten", "VERB", "Ich arbeite gern.")
        assert "Verb" in prompt
        assert "arbeiten" in prompt
        assert "Preterite" in prompt
        assert "past participle" in prompt.lower()

    def test_build_prompt_other(self):
        """Should build correct prompt for other POS."""
        enricher = Enricher()
        prompt = enricher._build_prompt("schnell", "ADJ", "Das Auto ist schnell.")
        assert "schnell" in prompt
        assert "base/dictionary form" in prompt

    def test_get_schema_for_noun(self):
        """Should return noun schema."""
        enricher = Enricher()
        schema, name = enricher._get_schema_for_pos("NOUN")
        assert name == "noun_enrichment"
        assert "gender" in schema.get("properties", {})
        assert "plural" in schema.get("properties", {})
        assert schema.get("additionalProperties") is False

    def test_get_schema_for_verb(self):
        """Should return verb schema."""
        enricher = Enricher()
        schema, name = enricher._get_schema_for_pos("VERB")
        assert name == "verb_enrichment"
        assert "preterite" in schema.get("properties", {})
        assert "auxiliary" in schema.get("properties", {})
        assert schema.get("additionalProperties") is False

    def test_get_schema_for_other(self):
        """Should return word schema for other POS."""
        enricher = Enricher()
        schema, name = enricher._get_schema_for_pos("ADJ")
        assert name == "word_enrichment"
        assert "lemma" in schema.get("properties", {})
        assert schema.get("additionalProperties") is False

    @pytest.mark.asyncio
    async def test_enrich_timeout(self):
        """Should handle timeout gracefully."""
        from openai import APITimeoutError

        enricher = Enricher()
        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = APITimeoutError(request=MagicMock())

            result = await enricher.enrich("test", "NOUN", "context")
            assert result.translations == []
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Should limit concurrent LLM tasks to 100."""
        # Check that semaphore is initialized with 100
        semaphore = get_semaphore()
        assert semaphore._value == 100

    @pytest.mark.asyncio
    async def test_validate_lemma_success(self):
        """Should validate lemma with LLM."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"valid": True, "corrected_lemma": "Arbeit", "reason": ""}

            result = await enricher.validate_lemma("Arbeit", "NOUN", "context")
            assert result["valid"] is True
            assert result["corrected_lemma"] == "Arbeit"

    @pytest.mark.asyncio
    async def test_validate_lemma_with_correction(self):
        """Should return corrected lemma when invalid."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "valid": False,
                "corrected_lemma": "Abbaustelle",
                "reason": "truncated word",
            }

            result = await enricher.validate_lemma("Abbaustell", "NOUN", "context")
            assert result["valid"] is False
            assert result["corrected_lemma"] == "Abbaustelle"
            assert result["reason"] == "truncated word"

    @pytest.mark.asyncio
    async def test_validate_lemma_connection_error(self):
        """Should handle connection errors gracefully."""
        from openai import APIConnectionError

        enricher = Enricher()
        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = APIConnectionError(request=MagicMock())

            result = await enricher.validate_lemma("Arbeit", "NOUN", "context")
            assert result["valid"] is True  # Default to valid on error
            assert result["corrected_lemma"] == "Arbeit"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_enrich_with_error_field(self):
        """Should set error field on failure."""
        from openai import APIConnectionError

        enricher = Enricher()
        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = APIConnectionError(request=MagicMock())

            result = await enricher.enrich("test", "NOUN", "context")
            assert result.error is not None
            assert "connect" in result.error.lower()

    @pytest.mark.asyncio
    async def test_enrich_success(self):
        """Should enrich word successfully."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "lemma": "Arbeit",
                "gender": "die",
                "plural": "Arbeiten",
                "translations": ["work"],
            }

            result = await enricher.enrich("Arbeit", "NOUN", "context")
            assert result.gender == "die"
            assert result.plural == "Arbeiten"
            assert result.translations == ["work"]
            assert result.error is None

    @pytest.mark.asyncio
    async def test_enrich_http_404_error(self):
        """Should handle 404 error with specific message."""
        from openai import APIStatusError

        enricher = Enricher()
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = APIStatusError("Not Found", response=mock_response, body=None)

            result = await enricher.enrich("test", "NOUN", "context")
            assert result.error is not None
            assert "not found" in result.error.lower()


class TestTextExtractor:
    """Tests for TextExtractor service."""

    def test_init_defaults(self):
        """Should use default whisper model."""
        extractor = TextExtractor()
        assert extractor.whisper_model == "large"
        assert extractor._whisper is None

    def test_init_custom_model(self):
        """Should accept custom whisper model."""
        extractor = TextExtractor(whisper_model="small")
        assert extractor.whisper_model == "small"

    def test_supported_extensions(self):
        """Should return all supported extensions."""
        extensions = TextExtractor.supported_extensions()
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".pdf" in extensions
        assert ".pptx" in extensions
        assert ".mp3" in extensions
        assert ".wav" in extensions
        assert ".m4a" in extensions

    @pytest.mark.asyncio
    async def test_extract_text_file(self, tmp_path: Path):
        """Should extract text from plain text files."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, World!", encoding="utf-8")

        extractor = TextExtractor()
        result = await extractor.extract(text_file)
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_extract_markdown_file(self, tmp_path: Path):
        """Should extract text from markdown files."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Heading\n\nParagraph text", encoding="utf-8")

        extractor = TextExtractor()
        result = await extractor.extract(md_file)
        assert "# Heading" in result
        assert "Paragraph text" in result

    @pytest.mark.asyncio
    async def test_extract_unsupported_format(self, tmp_path: Path):
        """Should raise error for unsupported formats."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("content")

        extractor = TextExtractor()
        with pytest.raises(ValueError, match="Unsupported file format"):
            await extractor.extract(unsupported)

    @pytest.mark.asyncio
    async def test_extract_pdf_sync(self, tmp_path: Path):
        """Should have sync PDF extraction method."""
        extractor = TextExtractor()
        # Just verify the method exists and is callable
        assert callable(extractor._extract_pdf_sync)

    @pytest.mark.asyncio
    async def test_extract_pptx_sync(self, tmp_path: Path):
        """Should have sync PPTX extraction method."""
        extractor = TextExtractor()
        # Just verify the method exists and is callable
        assert callable(extractor._extract_pptx_sync)

    @pytest.mark.asyncio
    async def test_extract_audio_sync(self, tmp_path: Path):
        """Should have sync audio transcription method."""
        extractor = TextExtractor()
        # Just verify the method exists and is callable
        assert callable(extractor._transcribe_audio_sync)
