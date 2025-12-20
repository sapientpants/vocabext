"""Tests for document processing tasks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, Word
from app.tasks.processing import _find_existing_word, process_document


class TestFindExistingWord:
    """Tests for _find_existing_word helper function."""

    @pytest.mark.asyncio
    async def test_finds_existing_word(self, async_session: AsyncSession):
        """Should find existing word by lemma and pos."""
        # Create a word
        word = Word(lemma="Arbeit", pos="NOUN")
        async_session.add(word)
        await async_session.commit()

        # Find it
        result = await _find_existing_word(async_session, "Arbeit", "NOUN")

        assert result is not None
        assert result.lemma == "Arbeit"
        assert result.pos == "NOUN"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, async_session: AsyncSession):
        """Should return None when word doesn't exist."""
        result = await _find_existing_word(async_session, "NonExistent", "NOUN")
        assert result is None

    @pytest.mark.asyncio
    async def test_matches_pos(self, async_session: AsyncSession):
        """Should only match when POS also matches."""
        # Create a noun
        word = Word(lemma="test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()

        # Look for verb with same lemma
        result = await _find_existing_word(async_session, "test", "VERB")
        assert result is None


class TestProcessDocument:
    """Tests for process_document function."""

    @pytest.mark.asyncio
    async def test_handles_missing_document(self):
        """Should handle missing document gracefully."""
        with patch("app.tasks.processing.async_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session.get.return_value = None
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            # Should not raise
            await process_document(99999)

    @pytest.mark.asyncio
    async def test_updates_status_on_extraction_error(self):
        """Should set error status when extraction fails."""
        with patch("app.tasks.processing.async_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_doc = MagicMock(spec=Document)
            mock_doc.id = 1
            mock_doc.filename = "test.pdf"
            mock_session.get.return_value = mock_doc
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            with patch("app.tasks.processing.TextExtractor") as mock_extractor:
                mock_instance = MagicMock()
                mock_instance.extract = AsyncMock(side_effect=Exception("Extraction failed"))
                mock_extractor.return_value = mock_instance

                await process_document(1)

                # Should have set error status
                assert mock_doc.status == "error"
                assert "Extraction failed" in mock_doc.error_message

    @pytest.mark.asyncio
    async def test_sets_pending_review_when_no_tokens(self):
        """Should set pending_review status when no tokens extracted."""
        with patch("app.tasks.processing.async_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_doc = MagicMock(spec=Document)
            mock_doc.id = 1
            mock_doc.filename = "test.txt"
            mock_session.get.return_value = mock_doc
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            with (
                patch("app.tasks.processing.TextExtractor") as mock_extractor,
                patch("app.tasks.processing.Tokenizer") as mock_tokenizer,
            ):
                mock_ext_instance = MagicMock()
                mock_ext_instance.extract = AsyncMock(return_value="Some text")
                mock_extractor.return_value = mock_ext_instance

                mock_tok_instance = MagicMock()
                mock_tok_instance.tokenize.return_value = []  # No tokens
                mock_tokenizer.return_value = mock_tok_instance

                await process_document(1)

                assert mock_doc.status == "pending_review"

    @pytest.mark.asyncio
    async def test_creates_duplicate_extraction_for_existing_word(self):
        """Should create duplicate extraction when word already exists."""
        with patch("app.tasks.processing.async_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_doc = MagicMock(spec=Document)
            mock_doc.id = 1
            mock_doc.filename = "test.txt"
            mock_session.get.return_value = mock_doc
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            existing_word = Word(
                id=1,
                lemma="Arbeit",
                pos="NOUN",
                gender="die",
            )

            with (
                patch("app.tasks.processing.TextExtractor") as mock_extractor,
                patch("app.tasks.processing.Tokenizer") as mock_tokenizer,
                patch(
                    "app.tasks.processing._find_existing_word",
                    new_callable=AsyncMock,
                    return_value=existing_word,
                ),
            ):
                mock_ext_instance = MagicMock()
                mock_ext_instance.extract = AsyncMock(return_value="Die Arbeit")
                mock_extractor.return_value = mock_ext_instance

                mock_token = MagicMock()
                mock_token.surface_form = "Arbeit"
                mock_token.lemma = "Arbeit"
                mock_token.pos = "NOUN"
                mock_token.context_sentence = "Die Arbeit ist wichtig."

                mock_tok_instance = MagicMock()
                mock_tok_instance.tokenize.return_value = [mock_token]
                mock_tokenizer.return_value = mock_tok_instance

                await process_document(1)

                # Should have added an extraction
                mock_session.add.assert_called()

    @pytest.mark.asyncio
    async def test_enriches_new_words(self):
        """Should enrich new words via LLM."""
        with patch("app.tasks.processing.async_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_doc = MagicMock(spec=Document)
            mock_doc.id = 1
            mock_doc.filename = "test.txt"
            mock_session.get.return_value = mock_doc
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            with (
                patch("app.tasks.processing.TextExtractor") as mock_extractor,
                patch("app.tasks.processing.Tokenizer") as mock_tokenizer,
                patch(
                    "app.tasks.processing._find_existing_word",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("app.tasks.processing.Enricher") as mock_enricher,
            ):
                mock_ext_instance = MagicMock()
                mock_ext_instance.extract = AsyncMock(return_value="Die Arbeit")
                mock_extractor.return_value = mock_ext_instance

                mock_token = MagicMock()
                mock_token.surface_form = "Arbeit"
                mock_token.lemma = "Arbeit"
                mock_token.pos = "NOUN"
                mock_token.context_sentence = "Die Arbeit."

                mock_tok_instance = MagicMock()
                mock_tok_instance.tokenize.return_value = [mock_token]
                mock_tokenizer.return_value = mock_tok_instance

                mock_enrichment = MagicMock()
                mock_enrichment.gender = "die"
                mock_enrichment.plural = "Arbeiten"
                mock_enrichment.translations = ["work"]
                mock_enrichment.preterite = None
                mock_enrichment.past_participle = None
                mock_enrichment.auxiliary = None

                mock_enr_instance = MagicMock()
                mock_enr_instance.enrich = AsyncMock(return_value=mock_enrichment)
                mock_enricher.return_value = mock_enr_instance

                await process_document(1)

                # Should have called enricher
                mock_enr_instance.enrich.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_enrichment_error(self):
        """Should handle enrichment errors gracefully."""
        with patch("app.tasks.processing.async_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_doc = MagicMock(spec=Document)
            mock_doc.id = 1
            mock_doc.filename = "test.txt"
            mock_session.get.return_value = mock_doc
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            with (
                patch("app.tasks.processing.TextExtractor") as mock_extractor,
                patch("app.tasks.processing.Tokenizer") as mock_tokenizer,
                patch(
                    "app.tasks.processing._find_existing_word",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch("app.tasks.processing.Enricher") as mock_enricher,
            ):
                mock_ext_instance = MagicMock()
                mock_ext_instance.extract = AsyncMock(return_value="Test")
                mock_extractor.return_value = mock_ext_instance

                mock_token = MagicMock()
                mock_token.surface_form = "Test"
                mock_token.lemma = "Test"
                mock_token.pos = "NOUN"
                mock_token.context_sentence = "Test."

                mock_tok_instance = MagicMock()
                mock_tok_instance.tokenize.return_value = [mock_token]
                mock_tokenizer.return_value = mock_tok_instance

                mock_enr_instance = MagicMock()
                mock_enr_instance.enrich = AsyncMock(side_effect=Exception("LLM error"))
                mock_enricher.return_value = mock_enr_instance

                # Should not raise
                await process_document(1)

                # Should still add extraction (without enrichment)
                mock_session.add.assert_called()

    @pytest.mark.asyncio
    async def test_handles_general_processing_error(self):
        """Should handle general processing errors."""
        with patch("app.tasks.processing.async_session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_doc = MagicMock(spec=Document)
            mock_doc.id = 1
            mock_doc.filename = "test.txt"

            # First get returns doc, second get (in error handler) also returns doc
            mock_session.get.return_value = mock_doc
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            with patch("app.tasks.processing.TextExtractor") as mock_extractor:
                mock_ext_instance = MagicMock()
                mock_ext_instance.extract = AsyncMock(return_value="Text")
                mock_extractor.return_value = mock_ext_instance

                with patch("app.tasks.processing.Tokenizer") as mock_tokenizer:
                    mock_tokenizer.side_effect = Exception("Tokenizer error")

                    # Should not raise
                    await process_document(1)

                    # Should have set error status
                    assert mock_doc.status == "error"
