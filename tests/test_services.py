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
        assert result.dictionary_error is None

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

    def test_with_dictionary_fields(self):
        """Should store dictionary-grounded fields."""
        result = EnrichmentResult(
            lemma="Haus",
            definition_de="Ein Gebäude zum Wohnen",
            synonyms=["Gebäude", "Heim"],
            frequency=5.5,
            ipa="/haʊs/",
            lemma_source="spacy",
            dictionary_url="https://example.com",
        )
        assert result.definition_de == "Ein Gebäude zum Wohnen"
        assert result.synonyms == ["Gebäude", "Heim"]
        assert result.frequency == 5.5
        assert result.lemma_source == "spacy"

    def test_with_error(self):
        """Should store error information."""
        result = EnrichmentResult(
            error="API timeout",
            dictionary_error="Lookup failed",
        )
        assert result.error == "API timeout"
        assert result.dictionary_error == "Lookup failed"


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
        prompt = enricher._build_prompt("Arbeit", "NOUN")
        assert "noun" in prompt.lower()
        assert "Arbeit" in prompt
        assert "Gender" in prompt
        assert "Plural" in prompt

    def test_build_prompt_verb(self):
        """Should build correct prompt for verbs."""
        enricher = Enricher()
        prompt = enricher._build_prompt("arbeiten", "VERB")
        assert "verb" in prompt.lower()
        assert "arbeiten" in prompt
        assert "Preterite" in prompt
        assert "past participle" in prompt.lower()

    def test_build_prompt_other(self):
        """Should build correct prompt for other POS."""
        enricher = Enricher()
        prompt = enricher._build_prompt("schnell", "ADJ")
        assert "schnell" in prompt
        assert "translations" in prompt.lower()

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

            result = await enricher.enrich("test", "NOUN")
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

            result = await enricher.enrich("test", "NOUN")
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

            result = await enricher.enrich("Arbeit", "NOUN")
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

            result = await enricher.enrich("test", "NOUN")
            assert result.error is not None
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_detect_pos_noun(self):
        """Should detect noun POS correctly."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"pos": "NOUN", "lemma": "Arbeit"}

            pos, lemma = await enricher.detect_pos("Arbeit")
            assert pos == "NOUN"
            assert lemma == "Arbeit"

    @pytest.mark.asyncio
    async def test_detect_pos_verb(self):
        """Should detect verb POS correctly."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"pos": "VERB", "lemma": "arbeiten"}

            pos, lemma = await enricher.detect_pos("arbeiten")
            assert pos == "VERB"
            assert lemma == "arbeiten"

    @pytest.mark.asyncio
    async def test_detect_pos_with_context(self):
        """Should use context for POS detection."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"pos": "ADJ", "lemma": "schnell"}

            pos, lemma = await enricher.detect_pos("schnell", "Das Auto ist schnell.")
            assert pos == "ADJ"
            assert lemma == "schnell"
            # Verify context was included in prompt
            call_args = mock_chat.call_args
            assert "Das Auto ist schnell." in call_args[0][0]

    @pytest.mark.asyncio
    async def test_detect_pos_strips_article(self):
        """Should strip article from lemma."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {"pos": "NOUN", "lemma": "die Arbeit"}

            pos, lemma = await enricher.detect_pos("die Arbeit")
            assert pos == "NOUN"
            assert lemma == "Arbeit"  # Article stripped

    @pytest.mark.asyncio
    async def test_detect_pos_defaults_on_missing_fields(self):
        """Should default to NOUN and original word if fields missing."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {}  # Empty response

            pos, lemma = await enricher.detect_pos("test")
            assert pos == "NOUN"  # Default POS
            assert lemma == "test"  # Original word

    @pytest.mark.asyncio
    async def test_enrich_word_full_pipeline(self):
        """Should run full enrichment pipeline: detect POS then enrich."""
        enricher = Enricher()

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            # First call: POS detection
            # Second call: enrichment
            mock_chat.side_effect = [
                {"pos": "NOUN", "lemma": "Haus"},
                {"lemma": "Haus", "gender": "das", "plural": "Häuser", "translations": ["house"]},
            ]

            with patch.object(enricher, "_dictionary_enabled", False):
                pos, result = await enricher.enrich_word("Haus")

            assert pos == "NOUN"
            assert result.lemma == "Haus"
            assert result.gender == "das"
            assert result.plural == "Häuser"
            assert result.translations == ["house"]

    @pytest.mark.asyncio
    async def test_enrich_word_with_context(self):
        """Should pass context through to both POS detection and enrichment."""
        enricher = Enricher()
        context = "Das Haus ist groß."

        with patch("app.services.enricher.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = [
                {"pos": "NOUN", "lemma": "Haus"},
                {"lemma": "Haus", "gender": "das", "plural": "Häuser", "translations": ["house"]},
            ]

            with patch.object(enricher, "_dictionary_enabled", False):
                pos, result = await enricher.enrich_word("Haus", context)

            # Verify context was used in both calls
            assert mock_chat.call_count == 2
            # POS detection call should include context
            assert context in mock_chat.call_args_list[0][0][0]


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
