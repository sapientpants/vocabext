"""Tests for vocabulary CLI commands."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.exceptions import Exit
from sqlalchemy import select

from app.cli.commands.vocabulary import _add_word
from app.models import Word, WordEvent
from app.services.enricher import EnrichmentResult
from app.services.tokenizer import TokenInfo


class TestAddWordCommand:
    """Tests for the vocab add command."""

    @pytest.mark.asyncio
    async def test_add_word_success(self, async_session):
        """Should add a word successfully with automatic POS detection."""
        # Mock TokenInfo from spaCy analysis
        mock_token_info = TokenInfo(
            surface_form="Haus",
            lemma="Haus",
            pos="NOUN",
            context_sentence="Das Haus ist groß.",
        )

        mock_enrichment = EnrichmentResult(
            lemma="Haus",
            gender="das",
            plural="Häuser",
            translations=["house", "home"],
            frequency=5.5,
            ipa="/haʊs/",
        )

        with (
            patch("app.cli.commands.vocabulary.Tokenizer") as MockTokenizer,
            patch("app.cli.commands.vocabulary.Enricher") as MockEnricher,
            patch("app.cli.commands.vocabulary.async_session") as mock_session_ctx,
            patch("app.cli.commands.vocabulary.is_model_loaded", return_value=True),
        ):
            # Setup mocks
            mock_tokenizer_instance = MockTokenizer.return_value
            mock_tokenizer_instance.analyze_word = MagicMock(return_value=mock_token_info)

            mock_enricher_instance = MockEnricher.return_value
            mock_enricher_instance.enrich_with_dictionary = AsyncMock(return_value=mock_enrichment)

            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=async_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            # Run the command - successful completion doesn't raise
            await _add_word("Haus", "")

        # Verify word was created
        stmt = select(Word).where(Word.lemma == "Haus")
        result = await async_session.execute(stmt)
        word = result.scalar_one_or_none()

        assert word is not None
        assert word.lemma == "Haus"
        assert word.pos == "NOUN"
        assert word.gender == "das"
        assert word.plural == "Häuser"
        assert word.translations_list == ["house", "home"]
        assert word.lemma_source == "spacy"

    @pytest.mark.asyncio
    async def test_add_word_duplicate_detection(self, async_session):
        """Should reject duplicate words."""
        # Create existing word
        existing = Word(
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
            plural="Arbeiten",
            translations=json.dumps(["work"]),
        )
        async_session.add(existing)
        await async_session.commit()

        # Mock TokenInfo from spaCy analysis
        mock_token_info = TokenInfo(
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            context_sentence="Die Arbeit ist wichtig.",
        )

        mock_enrichment = EnrichmentResult(
            lemma="Arbeit",
            gender="die",
            plural="Arbeiten",
            translations=["work"],
        )

        with (
            patch("app.cli.commands.vocabulary.Tokenizer") as MockTokenizer,
            patch("app.cli.commands.vocabulary.Enricher") as MockEnricher,
            patch("app.cli.commands.vocabulary.async_session") as mock_session_ctx,
            patch("app.cli.commands.vocabulary.is_model_loaded", return_value=True),
        ):
            mock_tokenizer_instance = MockTokenizer.return_value
            mock_tokenizer_instance.analyze_word = MagicMock(return_value=mock_token_info)

            mock_enricher_instance = MockEnricher.return_value
            mock_enricher_instance.enrich_with_dictionary = AsyncMock(return_value=mock_enrichment)

            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=async_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            # Should exit with code 1 for duplicate
            with pytest.raises(Exit) as exc_info:
                await _add_word("Arbeit", "")

            assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_add_word_empty_word_rejected(self, async_session):
        """Should reject empty word input."""
        with pytest.raises(Exit) as exc_info:
            await _add_word("", "")

        assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_add_word_whitespace_only_rejected(self, async_session):
        """Should reject whitespace-only word input."""
        with pytest.raises(Exit) as exc_info:
            await _add_word("   ", "")

        assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_add_word_non_alphabetic_rejected(self, async_session):
        """Should reject words with non-alphabetic characters."""
        with pytest.raises(Exit) as exc_info:
            await _add_word("Wort123", "")

        assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_add_word_with_hyphen_rejected(self, async_session):
        """Should reject words with hyphens."""
        with pytest.raises(Exit) as exc_info:
            await _add_word("Baden-Württemberg", "")

        assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_add_word_german_umlauts_accepted(self, async_session):
        """Should accept words with German umlauts."""
        # Mock TokenInfo from spaCy analysis
        mock_token_info = TokenInfo(
            surface_form="Größe",
            lemma="Größe",
            pos="NOUN",
            context_sentence="Die Größe ist wichtig.",
        )

        mock_enrichment = EnrichmentResult(
            lemma="Größe",
            gender="die",
            plural="Größen",
            translations=["size"],
        )

        with (
            patch("app.cli.commands.vocabulary.Tokenizer") as MockTokenizer,
            patch("app.cli.commands.vocabulary.Enricher") as MockEnricher,
            patch("app.cli.commands.vocabulary.async_session") as mock_session_ctx,
            patch("app.cli.commands.vocabulary.is_model_loaded", return_value=True),
        ):
            mock_tokenizer_instance = MockTokenizer.return_value
            mock_tokenizer_instance.analyze_word = MagicMock(return_value=mock_token_info)

            mock_enricher_instance = MockEnricher.return_value
            mock_enricher_instance.enrich_with_dictionary = AsyncMock(return_value=mock_enrichment)

            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=async_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            # Should not raise - umlauts are valid
            await _add_word("Größe", "")

    @pytest.mark.asyncio
    async def test_add_word_records_event(self, async_session):
        """Should record CREATED event for new word."""
        # Mock TokenInfo from spaCy analysis
        mock_token_info = TokenInfo(
            surface_form="Test",
            lemma="Test",
            pos="NOUN",
            context_sentence="Das ist ein Test.",
        )

        mock_enrichment = EnrichmentResult(
            lemma="Test",
            translations=["test"],
        )

        with (
            patch("app.cli.commands.vocabulary.Tokenizer") as MockTokenizer,
            patch("app.cli.commands.vocabulary.Enricher") as MockEnricher,
            patch("app.cli.commands.vocabulary.async_session") as mock_session_ctx,
            patch("app.cli.commands.vocabulary.is_model_loaded", return_value=True),
        ):
            mock_tokenizer_instance = MockTokenizer.return_value
            mock_tokenizer_instance.analyze_word = MagicMock(return_value=mock_token_info)

            mock_enricher_instance = MockEnricher.return_value
            mock_enricher_instance.enrich_with_dictionary = AsyncMock(return_value=mock_enrichment)

            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=async_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            try:
                await _add_word("Test", "")
            except SystemExit:
                pass

        # Verify event was created
        stmt = select(WordEvent).where(WordEvent.event_type == "CREATED")
        result = await async_session.execute(stmt)
        event = result.scalar_one_or_none()

        assert event is not None
        assert event.source == "cli"
        assert "vocab add" in event.reason

    @pytest.mark.asyncio
    async def test_add_word_with_context(self, async_session):
        """Should pass context to enricher."""
        # Mock TokenInfo from spaCy analysis
        mock_token_info = TokenInfo(
            surface_form="schnell",
            lemma="schnell",
            pos="ADJ",
            context_sentence="Das Auto ist schnell.",
        )

        mock_enrichment = EnrichmentResult(
            lemma="schnell",
            translations=["fast", "quick"],
        )
        context = "Das Auto ist schnell."

        with (
            patch("app.cli.commands.vocabulary.Tokenizer") as MockTokenizer,
            patch("app.cli.commands.vocabulary.Enricher") as MockEnricher,
            patch("app.cli.commands.vocabulary.async_session") as mock_session_ctx,
            patch("app.cli.commands.vocabulary.is_model_loaded", return_value=True),
        ):
            mock_tokenizer_instance = MockTokenizer.return_value
            mock_tokenizer_instance.analyze_word = MagicMock(return_value=mock_token_info)

            mock_enricher_instance = MockEnricher.return_value
            mock_enricher_instance.enrich_with_dictionary = AsyncMock(return_value=mock_enrichment)

            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=async_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            try:
                await _add_word("schnell", context)
            except SystemExit:
                pass

            # Verify context was passed to tokenizer
            mock_tokenizer_instance.analyze_word.assert_called_once_with("schnell", context)

    @pytest.mark.asyncio
    async def test_add_word_enrichment_error_marks_for_review(self, async_session):
        """Should mark word for review if enrichment has errors."""
        # Mock TokenInfo from spaCy analysis
        mock_token_info = TokenInfo(
            surface_form="Unknown",
            lemma="Unknown",
            pos="NOUN",
            context_sentence="Unknown",
        )

        mock_enrichment = EnrichmentResult(
            lemma="Unknown",
            translations=[],
            error="Dictionary lookup failed",
        )

        with (
            patch("app.cli.commands.vocabulary.Tokenizer") as MockTokenizer,
            patch("app.cli.commands.vocabulary.Enricher") as MockEnricher,
            patch("app.cli.commands.vocabulary.async_session") as mock_session_ctx,
            patch("app.cli.commands.vocabulary.is_model_loaded", return_value=True),
        ):
            mock_tokenizer_instance = MockTokenizer.return_value
            mock_tokenizer_instance.analyze_word = MagicMock(return_value=mock_token_info)

            mock_enricher_instance = MockEnricher.return_value
            mock_enricher_instance.enrich_with_dictionary = AsyncMock(return_value=mock_enrichment)

            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=async_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            try:
                await _add_word("Unknown", "")
            except SystemExit:
                pass

        # Verify word is marked for review
        stmt = select(Word).where(Word.lemma == "Unknown")
        result = await async_session.execute(stmt)
        word = result.scalar_one_or_none()

        assert word is not None
        assert word.needs_review is True
        assert word.review_reason == "Dictionary lookup failed"

    @pytest.mark.asyncio
    async def test_add_word_noun_duplicate_includes_gender(self, async_session):
        """Should include gender in duplicate check for nouns."""
        # Create existing word with different gender (hypothetical scenario)
        existing = Word(
            lemma="Band",
            pos="NOUN",
            gender="das",  # das Band = ribbon
            translations=json.dumps(["ribbon"]),
        )
        async_session.add(existing)
        await async_session.commit()

        # Mock TokenInfo from spaCy analysis
        mock_token_info = TokenInfo(
            surface_form="Band",
            lemma="Band",
            pos="NOUN",
            context_sentence="Die Band spielt heute.",
        )

        # Try to add same lemma but different gender
        mock_enrichment = EnrichmentResult(
            lemma="Band",
            gender="die",  # die Band = band/group
            translations=["band", "group"],
        )

        with (
            patch("app.cli.commands.vocabulary.Tokenizer") as MockTokenizer,
            patch("app.cli.commands.vocabulary.Enricher") as MockEnricher,
            patch("app.cli.commands.vocabulary.async_session") as mock_session_ctx,
            patch("app.cli.commands.vocabulary.is_model_loaded", return_value=True),
        ):
            mock_tokenizer_instance = MockTokenizer.return_value
            mock_tokenizer_instance.analyze_word = MagicMock(return_value=mock_token_info)

            mock_enricher_instance = MockEnricher.return_value
            mock_enricher_instance.enrich_with_dictionary = AsyncMock(return_value=mock_enrichment)

            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=async_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            try:
                await _add_word("Band", "")
            except SystemExit:
                pass

        # Should have added the new word (different gender = different word)
        stmt = select(Word).where(Word.lemma == "Band", Word.gender == "die")
        result = await async_session.execute(stmt)
        new_word = result.scalar_one_or_none()

        assert new_word is not None
        assert new_word.gender == "die"
