"""Tests for SQLAlchemy models."""

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, Extraction, Word, _utc_now


class TestUtcNow:
    """Tests for _utc_now helper function."""

    def test_returns_datetime(self):
        """Should return a datetime object."""
        result = _utc_now()
        assert isinstance(result, datetime)

    def test_returns_utc_timezone(self):
        """Should return a datetime with UTC timezone."""
        result = _utc_now()
        assert result.tzinfo == timezone.utc

    def test_returns_current_time(self):
        """Should return approximately the current time."""
        before = datetime.now(timezone.utc)
        result = _utc_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


class TestDocument:
    """Tests for Document model."""

    @pytest.mark.asyncio
    async def test_create_document(self, async_session: AsyncSession):
        """Should create a document successfully."""
        doc = Document(
            filename="test.pdf",
            content_hash="abc123",
            status="processing",
        )
        async_session.add(doc)
        await async_session.commit()

        assert doc.id is not None
        assert doc.filename == "test.pdf"
        assert doc.content_hash == "abc123"
        assert doc.status == "processing"
        assert doc.created_at is not None

    @pytest.mark.asyncio
    async def test_document_pending_count(self, async_session: AsyncSession):
        """Should correctly count pending extractions."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        # Add extractions with different statuses
        for status in ["pending", "pending", "accepted", "rejected"]:
            extraction = Extraction(
                document_id=doc.id,
                surface_form="test",
                lemma="test",
                pos="NOUN",
                status=status,
            )
            async_session.add(extraction)
        await async_session.commit()

        # Refresh to load the relationship
        await async_session.refresh(doc, ["extractions"])
        assert doc.pending_count == 2

    @pytest.mark.asyncio
    async def test_document_duplicate_count(self, async_session: AsyncSession):
        """Should correctly count duplicate extractions."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        # Add extractions with different statuses
        for status in ["duplicate", "duplicate", "duplicate", "pending"]:
            extraction = Extraction(
                document_id=doc.id,
                surface_form="test",
                lemma="test",
                pos="NOUN",
                status=status,
            )
            async_session.add(extraction)
        await async_session.commit()

        # Refresh to load the relationship
        await async_session.refresh(doc, ["extractions"])
        assert doc.duplicate_count == 3


class TestWord:
    """Tests for Word model."""

    @pytest.mark.asyncio
    async def test_create_word(self, async_session: AsyncSession, sample_word_data):
        """Should create a word successfully."""
        word = Word(**sample_word_data)
        async_session.add(word)
        await async_session.commit()

        assert word.id is not None
        assert word.lemma == "Arbeit"
        assert word.pos == "NOUN"
        assert word.gender == "die"

    def test_translations_list_getter(self):
        """Should parse JSON translations to list."""
        word = Word(
            lemma="test",
            pos="NOUN",
            translations='["one", "two", "three"]',
        )
        assert word.translations_list == ["one", "two", "three"]

    def test_translations_list_getter_empty(self):
        """Should return empty list when no translations."""
        word = Word(lemma="test", pos="NOUN", translations=None)
        assert word.translations_list == []

    def test_translations_list_getter_invalid_json(self):
        """Should return empty list for invalid JSON."""
        word = Word(lemma="test", pos="NOUN", translations="not json")
        assert word.translations_list == []

    def test_translations_list_setter(self):
        """Should serialize list to JSON."""
        word = Word(lemma="test", pos="NOUN")
        word.translations_list = ["a", "b", "c"]
        assert word.translations == '["a", "b", "c"]'

    def test_display_word_noun_with_gender(self):
        """Should include article for nouns with gender."""
        word = Word(lemma="Arbeit", pos="NOUN", gender="die")
        assert word.display_word == "die Arbeit"

    def test_display_word_noun_without_gender(self):
        """Should return just lemma for nouns without gender."""
        word = Word(lemma="Test", pos="NOUN", gender=None)
        assert word.display_word == "Test"

    def test_display_word_verb(self):
        """Should return just lemma for verbs."""
        word = Word(lemma="arbeiten", pos="VERB")
        assert word.display_word == "arbeiten"

    def test_grammar_info_noun(self):
        """Should format plural info for nouns."""
        word = Word(lemma="Arbeit", pos="NOUN", plural="Arbeiten")
        assert word.grammar_info == "pl: Arbeiten"

    def test_grammar_info_verb(self):
        """Should format verb conjugation info."""
        word = Word(
            lemma="arbeiten",
            pos="VERB",
            preterite="arbeitete",
            past_participle="gearbeitet",
            auxiliary="haben",
        )
        assert "prät: arbeitete" in word.grammar_info
        assert "pp: haben gearbeitet" in word.grammar_info

    def test_grammar_info_verb_default_auxiliary(self):
        """Should default to 'haben' for auxiliary."""
        word = Word(
            lemma="arbeiten",
            pos="VERB",
            past_participle="gearbeitet",
        )
        assert "pp: haben gearbeitet" in word.grammar_info

    def test_grammar_info_adjective(self):
        """Should return empty for adjectives."""
        word = Word(lemma="schnell", pos="ADJ")
        assert word.grammar_info == ""

    def test_grammar_info_adverb(self):
        """Should return empty for adverbs."""
        word = Word(lemma="schnell", pos="ADV")
        assert word.grammar_info == ""

    def test_is_synced_true(self):
        """Should return True when anki_note_id is set."""
        word = Word(lemma="test", pos="NOUN", anki_note_id=12345)
        assert word.is_synced is True

    def test_is_synced_false(self):
        """Should return False when anki_note_id is None."""
        word = Word(lemma="test", pos="NOUN", anki_note_id=None)
        assert word.is_synced is False


class TestExtraction:
    """Tests for Extraction model."""

    @pytest.mark.asyncio
    async def test_create_extraction(self, async_session: AsyncSession, sample_extraction_data):
        """Should create an extraction successfully."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        extraction = Extraction(document_id=doc.id, **sample_extraction_data)
        async_session.add(extraction)
        await async_session.commit()

        assert extraction.id is not None
        assert extraction.document_id == doc.id
        assert extraction.status == "pending"

    def test_extraction_display_word(self):
        """Should format display word correctly."""
        extraction = Extraction(
            document_id=1,
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
        )
        assert extraction.display_word == "die Arbeit"

    def test_extraction_display_word_no_gender(self):
        """Should format display word without gender."""
        extraction = Extraction(
            document_id=1,
            surface_form="test",
            lemma="Test",
            pos="NOUN",
            gender=None,
        )
        assert extraction.display_word == "Test"

    def test_extraction_display_word_verb(self):
        """Should format verb display word."""
        extraction = Extraction(
            document_id=1,
            surface_form="arbeitet",
            lemma="arbeiten",
            pos="VERB",
        )
        assert extraction.display_word == "arbeiten"

    def test_extraction_translations_list(self):
        """Should parse translations list correctly."""
        extraction = Extraction(
            document_id=1,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            translations='["a", "b"]',
        )
        assert extraction.translations_list == ["a", "b"]

    def test_extraction_translations_list_empty(self):
        """Should return empty list for no translations."""
        extraction = Extraction(
            document_id=1,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            translations=None,
        )
        assert extraction.translations_list == []

    def test_extraction_translations_list_invalid(self):
        """Should return empty list for invalid JSON."""
        extraction = Extraction(
            document_id=1,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            translations="not json",
        )
        assert extraction.translations_list == []

    def test_extraction_translations_list_setter(self):
        """Should set translations from list."""
        extraction = Extraction(
            document_id=1,
            surface_form="test",
            lemma="test",
            pos="NOUN",
        )
        extraction.translations_list = ["a", "b", "c"]
        assert extraction.translations == '["a", "b", "c"]'

    def test_extraction_grammar_info_verb(self):
        """Should format verb grammar info."""
        extraction = Extraction(
            document_id=1,
            surface_form="arbeitete",
            lemma="arbeiten",
            pos="VERB",
            preterite="arbeitete",
            past_participle="gearbeitet",
            auxiliary="haben",
        )
        assert "prät: arbeitete" in extraction.grammar_info
        assert "pp: haben gearbeitet" in extraction.grammar_info

    def test_extraction_grammar_info_noun(self):
        """Should format noun grammar info."""
        extraction = Extraction(
            document_id=1,
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            plural="Arbeiten",
        )
        assert "pl: Arbeiten" in extraction.grammar_info
