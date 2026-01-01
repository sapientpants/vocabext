"""Tests for dictionary services."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.dictionary.base import DictionaryEntry
from app.services.dictionary.cache import (
    CacheManager,
    _deserialize_entry,
    _serialize_entry,
)
from app.services.dictionary.service import DictionaryService
from app.services.dictionary.spacy_backend import (
    SpacyBackend,
    is_model_loaded,
    preload_model,
)


class TestSpacyBackendInit:
    """Tests for SpacyBackend initialization."""

    def test_init_without_nlp(self):
        """Should initialize without nlp model."""
        backend = SpacyBackend()
        assert backend._nlp is None

    def test_init_with_nlp(self):
        """Should use provided nlp model."""
        mock_nlp = MagicMock()
        backend = SpacyBackend(nlp=mock_nlp)
        assert backend._nlp is mock_nlp

    def test_name_property(self):
        """Should return 'spacy' as name."""
        backend = SpacyBackend()
        assert backend.name == "spacy"


class TestSpacyBackendNlpProperty:
    """Tests for SpacyBackend.nlp property."""

    def test_nlp_lazy_loads_model(self):
        """Should lazy-load spaCy model when accessed."""
        mock_nlp = MagicMock()
        with patch("app.services.dictionary.spacy_backend._get_nlp", return_value=mock_nlp):
            backend = SpacyBackend()
            nlp = backend.nlp
            assert nlp is mock_nlp

    def test_nlp_uses_provided_model(self):
        """Should use provided model instead of loading."""
        mock_nlp = MagicMock()
        backend = SpacyBackend(nlp=mock_nlp)
        assert backend.nlp is mock_nlp


class TestSpacyBackendIsKnown:
    """Tests for SpacyBackend._is_known method."""

    def test_known_word_returns_true(self):
        """Should return True for known words."""
        mock_nlp = MagicMock()
        mock_lexeme = MagicMock()
        mock_lexeme.is_oov = False
        mock_nlp.vocab.__getitem__.return_value = mock_lexeme

        backend = SpacyBackend(nlp=mock_nlp)
        assert backend._is_known("Haus") is True

    def test_unknown_word_returns_false(self):
        """Should return False for unknown words."""
        mock_nlp = MagicMock()
        mock_lexeme = MagicMock()
        mock_lexeme.is_oov = True
        mock_nlp.vocab.__getitem__.return_value = mock_lexeme

        backend = SpacyBackend(nlp=mock_nlp)
        assert backend._is_known("xyzabc123") is False


class TestSpacyBackendLookup:
    """Tests for SpacyBackend.lookup method."""

    @pytest.mark.asyncio
    async def test_lookup_known_word(self):
        """Should return entry for known word."""
        mock_nlp = MagicMock()
        mock_lexeme = MagicMock()
        mock_lexeme.is_oov = False
        mock_nlp.vocab.__getitem__.return_value = mock_lexeme

        backend = SpacyBackend(nlp=mock_nlp)
        result = await backend.lookup("Haus", "NOUN")

        assert result is not None
        assert result.lemma == "Haus"
        assert result.source == "spacy"
        assert result.lemma_validated is True

    @pytest.mark.asyncio
    async def test_lookup_unknown_word_tries_lowercase(self):
        """Should try lowercase for unknown words."""
        mock_nlp = MagicMock()

        def vocab_lookup(word):
            lexeme = MagicMock()
            # "HAUS" is unknown, "haus" is known
            lexeme.is_oov = word == "HAUS"
            return lexeme

        mock_nlp.vocab.__getitem__.side_effect = vocab_lookup

        backend = SpacyBackend(nlp=mock_nlp)
        result = await backend.lookup("HAUS")

        assert result is not None
        assert result.lemma == "haus"

    @pytest.mark.asyncio
    async def test_lookup_unknown_word_returns_none(self):
        """Should return None for completely unknown words."""
        mock_nlp = MagicMock()
        mock_lexeme = MagicMock()
        mock_lexeme.is_oov = True
        mock_nlp.vocab.__getitem__.return_value = mock_lexeme

        backend = SpacyBackend(nlp=mock_nlp)
        result = await backend.lookup("xyzabc123")

        assert result is None


class TestSpacyBackendValidateLemma:
    """Tests for SpacyBackend.validate_lemma method."""

    @pytest.mark.asyncio
    async def test_validate_known_word(self):
        """Should return True, None for known words."""
        mock_nlp = MagicMock()
        mock_lexeme = MagicMock()
        mock_lexeme.is_oov = False
        mock_nlp.vocab.__getitem__.return_value = mock_lexeme

        backend = SpacyBackend(nlp=mock_nlp)
        is_valid, correction = await backend.validate_lemma("Haus")

        assert is_valid is True
        assert correction is None

    @pytest.mark.asyncio
    async def test_validate_corrects_case(self):
        """Should correct case for known lowercase forms."""
        mock_nlp = MagicMock()

        def vocab_lookup(word):
            lexeme = MagicMock()
            # "HAUS" unknown, "haus" known
            lexeme.is_oov = word == "HAUS"
            return lexeme

        mock_nlp.vocab.__getitem__.side_effect = vocab_lookup

        backend = SpacyBackend(nlp=mock_nlp)
        is_valid, correction = await backend.validate_lemma("HAUS")

        assert is_valid is True
        assert correction == "haus"

    @pytest.mark.asyncio
    async def test_validate_tries_title_case_for_nouns(self):
        """Should try title case for nouns."""
        mock_nlp = MagicMock()

        def vocab_lookup(word):
            lexeme = MagicMock()
            # "haus" unknown, "Haus" known
            lexeme.is_oov = word != "Haus"
            return lexeme

        mock_nlp.vocab.__getitem__.side_effect = vocab_lookup

        backend = SpacyBackend(nlp=mock_nlp)
        is_valid, correction = await backend.validate_lemma("haus", "NOUN")

        assert is_valid is True
        assert correction == "Haus"

    @pytest.mark.asyncio
    async def test_validate_unknown_word_assumed_valid(self):
        """Unknown words should be assumed valid."""
        mock_nlp = MagicMock()
        mock_lexeme = MagicMock()
        mock_lexeme.is_oov = True
        mock_nlp.vocab.__getitem__.return_value = mock_lexeme

        backend = SpacyBackend(nlp=mock_nlp)
        is_valid, correction = await backend.validate_lemma("xyzabc123")

        assert is_valid is True
        assert correction is None


class TestSpacyModuleFunctions:
    """Tests for module-level spaCy functions."""

    def test_is_model_loaded_false_initially(self):
        """Should return False when model not loaded."""
        with patch("app.services.dictionary.spacy_backend._nlp", None):
            assert is_model_loaded() is False

    def test_is_model_loaded_true_after_load(self):
        """Should return True when model is loaded."""
        with patch("app.services.dictionary.spacy_backend._nlp", MagicMock()):
            assert is_model_loaded() is True

    def test_preload_model_calls_get_nlp(self):
        """preload_model should trigger model loading."""
        with patch("app.services.dictionary.spacy_backend._get_nlp") as mock_get:
            preload_model()
            mock_get.assert_called_once()


class TestDictionaryEntry:
    """Tests for DictionaryEntry dataclass."""

    def test_create_minimal_entry(self):
        """Should create entry with minimal fields."""
        entry = DictionaryEntry(lemma="Test", source="test")
        assert entry.lemma == "Test"
        assert entry.source == "test"
        assert entry.gender is None
        assert entry.pos is None

    def test_create_full_entry(self):
        """Should create entry with all fields."""
        entry = DictionaryEntry(
            lemma="Haus",
            pos="NOUN",
            gender="das",
            definitions=["house", "building"],
            synonyms=["Gebäude"],
            frequency=5.5,
            ipa="/haʊs/",
            url="https://example.com",
            source="test",
            lemma_validated=True,
        )
        assert entry.lemma == "Haus"
        assert entry.gender == "das"
        assert entry.definitions == ["house", "building"]
        assert entry.frequency == 5.5


class TestDictionaryService:
    """Tests for DictionaryService class."""

    def test_init_defaults(self):
        """Should initialize with default backends."""
        service = DictionaryService()
        assert len(service.backends) == 1
        assert service.use_cache is True

    def test_init_custom_backends(self):
        """Should use provided backends."""
        mock_backend = MagicMock()
        service = DictionaryService(backends=[mock_backend])
        assert service.backends == [mock_backend]

    def test_init_no_cache(self):
        """Should allow disabling cache."""
        service = DictionaryService(use_cache=False)
        assert service.use_cache is False

    @pytest.mark.asyncio
    async def test_lookup_returns_first_result(self):
        """Should return first successful backend result."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.lookup = AsyncMock(return_value=DictionaryEntry(lemma="Test", source="test"))

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        result = await service.lookup("Test")

        assert result is not None
        assert result.lemma == "Test"

    @pytest.mark.asyncio
    async def test_lookup_tries_multiple_backends(self):
        """Should try next backend if first returns None."""
        backend1 = MagicMock()
        backend1.name = "backend1"
        backend1.lookup = AsyncMock(return_value=None)

        backend2 = MagicMock()
        backend2.name = "backend2"
        backend2.lookup = AsyncMock(return_value=DictionaryEntry(lemma="Test", source="backend2"))

        service = DictionaryService(backends=[backend1, backend2], use_cache=False)
        result = await service.lookup("Test")

        assert result is not None
        assert result.source == "backend2"

    @pytest.mark.asyncio
    async def test_lookup_handles_timeout(self):
        """Should handle backend timeouts gracefully."""
        mock_backend = MagicMock()
        mock_backend.name = "slow"
        mock_backend.lookup = AsyncMock(side_effect=asyncio.TimeoutError())

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        result = await service.lookup("Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_handles_exception(self):
        """Should handle backend exceptions gracefully."""
        mock_backend = MagicMock()
        mock_backend.name = "broken"
        mock_backend.lookup = AsyncMock(side_effect=Exception("Backend error"))

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        result = await service.lookup("Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_and_ground_lemma_success(self):
        """Should validate lemma against backends."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.validate_lemma = AsyncMock(return_value=(True, None))

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        lemma, is_grounded, source = await service.validate_and_ground_lemma("Haus", "NOUN")

        assert lemma == "Haus"
        assert is_grounded is True
        assert source == "test"

    @pytest.mark.asyncio
    async def test_validate_and_ground_lemma_with_correction(self):
        """Should return corrected lemma."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.validate_lemma = AsyncMock(return_value=(True, "Corrected"))

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        lemma, is_grounded, source = await service.validate_and_ground_lemma("wrong", "NOUN")

        assert lemma == "Corrected"
        assert is_grounded is True

    @pytest.mark.asyncio
    async def test_validate_and_ground_lemma_not_found(self):
        """Should return original lemma if not validated."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.validate_lemma = AsyncMock(return_value=(False, None))

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        lemma, is_grounded, source = await service.validate_and_ground_lemma("unknown", "NOUN")

        assert lemma == "unknown"
        assert is_grounded is False
        assert source is None

    @pytest.mark.asyncio
    async def test_validate_handles_timeout(self):
        """Should handle validation timeout."""
        mock_backend = MagicMock()
        mock_backend.name = "slow"
        mock_backend.validate_lemma = AsyncMock(side_effect=asyncio.TimeoutError())

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        lemma, is_grounded, source = await service.validate_and_ground_lemma("Test", "NOUN")

        assert lemma == "Test"
        assert is_grounded is False

    @pytest.mark.asyncio
    async def test_validate_handles_exception(self):
        """Should handle validation exception."""
        mock_backend = MagicMock()
        mock_backend.name = "broken"
        mock_backend.validate_lemma = AsyncMock(side_effect=Exception("Error"))

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        lemma, is_grounded, source = await service.validate_and_ground_lemma("Test", "NOUN")

        assert is_grounded is False

    @pytest.mark.asyncio
    async def test_get_enrichment_data_delegates_to_lookup(self):
        """get_enrichment_data should delegate to lookup."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_backend.lookup = AsyncMock(return_value=DictionaryEntry(lemma="Test", source="test"))

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        result = await service.get_enrichment_data("Test", "NOUN")

        assert result is not None
        mock_backend.lookup.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_calls_backend_close(self):
        """close should call close on backends that have it."""
        mock_backend = MagicMock()
        mock_backend.close = AsyncMock()

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        await service.close()

        mock_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_backends_without_close(self):
        """close should handle backends without close method."""
        mock_backend = MagicMock(spec=[])  # No close method

        service = DictionaryService(backends=[mock_backend], use_cache=False)
        await service.close()  # Should not raise


class TestCacheSerialization:
    """Tests for cache serialization functions."""

    def test_serialize_entry(self):
        """Should serialize DictionaryEntry to JSON."""
        entry = DictionaryEntry(
            lemma="Haus",
            pos="NOUN",
            gender="das",
            definitions=["house"],
            source="test",
        )
        json_str = _serialize_entry(entry)
        assert '"lemma": "Haus"' in json_str
        assert '"gender": "das"' in json_str

    def test_deserialize_entry(self):
        """Should deserialize JSON to DictionaryEntry."""
        json_str = '{"lemma": "Haus", "pos": "NOUN", "gender": "das", "source": "test"}'
        entry = _deserialize_entry(json_str)
        assert entry.lemma == "Haus"
        assert entry.pos == "NOUN"
        assert entry.gender == "das"

    def test_serialize_deserialize_roundtrip(self):
        """Should preserve data through serialize/deserialize."""
        original = DictionaryEntry(
            lemma="Test",
            pos="VERB",
            definitions=["to test"],
            synonyms=["prüfen"],
            frequency=4.5,
            source="test",
        )
        json_str = _serialize_entry(original)
        restored = _deserialize_entry(json_str)

        assert restored.lemma == original.lemma
        assert restored.pos == original.pos
        assert restored.definitions == original.definitions
        assert restored.synonyms == original.synonyms
        assert restored.frequency == original.frequency


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init_default_ttl(self):
        """Should use default TTL."""
        manager = CacheManager()
        assert manager.ttl_days == CacheManager.DEFAULT_TTL_DAYS

    def test_init_custom_ttl(self):
        """Should use custom TTL."""
        manager = CacheManager(ttl_days=60)
        assert manager.ttl_days == 60

    def test_init_custom_not_found_ttl(self):
        """Should use custom not-found TTL."""
        manager = CacheManager(not_found_ttl_days=14)
        assert manager.not_found_ttl_days == 14
