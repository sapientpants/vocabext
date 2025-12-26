"""Tests for service modules."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enricher import Enricher, EnrichmentResult, _llm_semaphore
from app.services.extractor import TextExtractor


class TestEnrichmentResult:
    """Tests for EnrichmentResult dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        result = EnrichmentResult()
        assert result.gender is None
        assert result.plural is None
        assert result.preterite is None
        assert result.past_participle is None
        assert result.auxiliary is None
        assert result.translations == []

    def test_post_init_none_translations(self):
        """Should convert None translations to empty list."""
        result = EnrichmentResult(translations=None)
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
        from app.config import settings

        enricher = Enricher()
        assert enricher.base_url == settings.ollama_base_url
        assert enricher.model == settings.ollama_model

    def test_init_custom(self):
        """Should accept custom settings."""
        enricher = Enricher(base_url="http://custom:1234", model="custom-model")
        assert enricher.base_url == "http://custom:1234"
        assert enricher.model == "custom-model"

    def test_build_prompt_noun(self):
        """Should build correct prompt for nouns."""
        enricher = Enricher()
        prompt = enricher._build_prompt("Arbeit", "NOUN", "Die Arbeit ist wichtig.")
        assert "Noun" in prompt
        assert "Arbeit" in prompt
        assert "gender" in prompt
        assert "plural" in prompt

    def test_build_prompt_verb(self):
        """Should build correct prompt for verbs."""
        enricher = Enricher()
        prompt = enricher._build_prompt("arbeiten", "VERB", "Ich arbeite gern.")
        assert "Verb" in prompt
        assert "arbeiten" in prompt
        assert "preterite" in prompt
        assert "past_participle" in prompt

    def test_build_prompt_other(self):
        """Should build correct prompt for other POS."""
        enricher = Enricher()
        prompt = enricher._build_prompt("schnell", "ADJ", "Das Auto ist schnell.")
        assert "schnell" in prompt
        assert "translations" in prompt

    def test_clean_json_removes_comments(self):
        """Should remove C-style comments from JSON."""
        enricher = Enricher()
        dirty = '{"key": "value" /* comment */}'
        clean = enricher._clean_json(dirty)
        assert "comment" not in clean
        # Note: whitespace where the comment was may remain
        assert "/* " not in clean
        assert " */" not in clean

    def test_clean_json_removes_trailing_commas(self):
        """Should remove trailing commas."""
        enricher = Enricher()
        dirty = '{"key": "value",}'
        clean = enricher._clean_json(dirty)
        assert '{"key": "value"}' == clean.strip()

    def test_parse_response_noun(self):
        """Should parse noun response correctly."""
        enricher = Enricher()
        response = '{"gender": "die", "plural": "Arbeiten", "translations": ["work"]}'
        result = enricher._parse_response(response, "NOUN")
        assert result.gender == "die"
        assert result.plural == "Arbeiten"
        assert result.translations == ["work"]

    def test_parse_response_verb(self):
        """Should parse verb response correctly."""
        enricher = Enricher()
        response = '{"preterite": "arbeitete", "past_participle": "gearbeitet", "auxiliary": "haben", "translations": ["to work"]}'
        result = enricher._parse_response(response, "VERB")
        assert result.preterite == "arbeitete"
        assert result.past_participle == "gearbeitet"
        assert result.auxiliary == "haben"

    def test_parse_response_markdown_code_block(self):
        """Should extract JSON from markdown code blocks."""
        enricher = Enricher()
        response = '```json\n{"translations": ["test"]}\n```'
        result = enricher._parse_response(response, "ADJ")
        assert result.translations == ["test"]

    def test_parse_response_invalid_json(self):
        """Should return empty result for invalid JSON."""
        enricher = Enricher()
        result = enricher._parse_response("not valid json", "NOUN")
        assert result.translations == []

    def test_parse_response_no_json(self):
        """Should return empty result when no JSON found."""
        enricher = Enricher()
        result = enricher._parse_response("just plain text", "NOUN")
        assert result.translations == []

    @pytest.mark.asyncio
    async def test_enrich_timeout(self):
        """Should handle timeout gracefully."""
        enricher = Enricher()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            import httpx

            mock_instance.post.side_effect = httpx.TimeoutException("timeout")

            result = await enricher.enrich("test", "NOUN", "context")
            assert result.translations == []

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Should only allow one LLM request at a time."""
        # Check that semaphore is initialized with 1
        assert _llm_semaphore._value == 1

    def test_extract_json_object_simple(self):
        """Should extract simple JSON object."""
        enricher = Enricher()
        text = 'Some text {"key": "value"} more text'
        result = enricher._extract_json_object(text)
        assert result == '{"key": "value"}'

    def test_extract_json_object_nested(self):
        """Should extract nested JSON object."""
        enricher = Enricher()
        text = '{"outer": {"inner": "value"}} trailing'
        result = enricher._extract_json_object(text)
        assert result == '{"outer": {"inner": "value"}}'

    def test_extract_json_object_with_strings(self):
        """Should handle braces inside strings."""
        enricher = Enricher()
        text = '{"key": "value with { brace }"} more'
        result = enricher._extract_json_object(text)
        assert result == '{"key": "value with { brace }"}'

    def test_extract_json_object_no_json(self):
        """Should return None when no JSON found."""
        enricher = Enricher()
        result = enricher._extract_json_object("no json here")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_lemma_success(self):
        """Should validate lemma with LLM."""
        enricher = Enricher()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": '{"valid": true, "corrected_lemma": "Arbeit"}'
            }
            mock_response.raise_for_status = MagicMock()
            mock_instance.post.return_value = mock_response

            result = await enricher.validate_lemma("Arbeit", "NOUN", "context")
            assert result["valid"] is True
            assert result["corrected_lemma"] == "Arbeit"

    @pytest.mark.asyncio
    async def test_validate_lemma_with_correction(self):
        """Should return corrected lemma when invalid."""
        enricher = Enricher()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": '{"valid": false, "corrected_lemma": "Abbaustelle", "reason": "truncated word"}'
            }
            mock_response.raise_for_status = MagicMock()
            mock_instance.post.return_value = mock_response

            result = await enricher.validate_lemma("Abbaustell", "NOUN", "context")
            assert result["valid"] is False
            assert result["corrected_lemma"] == "Abbaustelle"
            assert result["reason"] == "truncated word"

    @pytest.mark.asyncio
    async def test_validate_lemma_connection_error(self):
        """Should handle connection errors gracefully."""
        enricher = Enricher()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            import httpx

            mock_instance.post.side_effect = httpx.ConnectError("connection failed")

            result = await enricher.validate_lemma("Arbeit", "NOUN", "context")
            assert result["valid"] is True  # Default to valid on error
            assert result["corrected_lemma"] == "Arbeit"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_enrich_with_error_field(self):
        """Should set error field on failure."""
        enricher = Enricher()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            import httpx

            mock_instance.post.side_effect = httpx.ConnectError("connection failed")

            result = await enricher.enrich("test", "NOUN", "context")
            assert result.error is not None
            assert "connect" in result.error.lower()

    @pytest.mark.asyncio
    async def test_enrich_success(self):
        """Should enrich word successfully."""
        enricher = Enricher()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": '{"lemma": "Arbeit", "gender": "die", "plural": "Arbeiten", "translations": ["work"]}'
            }
            mock_response.raise_for_status = MagicMock()
            mock_instance.post.return_value = mock_response

            result = await enricher.enrich("Arbeit", "NOUN", "context")
            assert result.gender == "die"
            assert result.plural == "Arbeiten"
            assert result.translations == ["work"]
            assert result.error is None

    @pytest.mark.asyncio
    async def test_enrich_http_404_error(self):
        """Should handle 404 error with specific message."""
        enricher = Enricher()
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            import httpx

            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_instance.post.side_effect = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )

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
