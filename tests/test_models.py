"""Tests for SQLAlchemy models."""

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Word, _utc_now


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

    def test_cases_list_getter(self):
        """Should parse JSON cases to list for prepositions."""
        word = Word(
            lemma="mit",
            pos="ADP",
            cases='["dativ"]',
        )
        assert word.cases_list == ["dativ"]

    def test_cases_list_getter_empty(self):
        """Should return empty list when no cases."""
        word = Word(lemma="mit", pos="ADP", cases=None)
        assert word.cases_list == []

    def test_cases_list_getter_invalid_json(self):
        """Should return empty list for invalid JSON."""
        word = Word(lemma="mit", pos="ADP", cases="not json")
        assert word.cases_list == []

    def test_cases_list_setter(self):
        """Should serialize cases list to JSON."""
        word = Word(lemma="mit", pos="ADP")
        word.cases_list = ["dativ", "akkusativ"]
        assert word.cases == '["dativ", "akkusativ"]'

    def test_grammar_info_preposition(self):
        """Should format case info for prepositions."""
        word = Word(lemma="mit", pos="ADP", cases='["dativ"]')
        assert word.grammar_info == "+ dativ"

    def test_grammar_info_preposition_multiple_cases(self):
        """Should format multiple cases for two-way prepositions."""
        word = Word(lemma="in", pos="ADP", cases='["akkusativ", "dativ"]')
        assert word.grammar_info == "+ akkusativ, dativ"


class TestWordVersionTranslationsList:
    """Tests for WordVersion.translations_list property."""

    def test_translations_list_valid_json(self):
        """Should parse valid JSON array."""
        from app.models import WordVersion

        version = WordVersion(
            word_id=1,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            translations='["one", "two"]',
        )
        assert version.translations_list == ["one", "two"]

    def test_translations_list_empty(self):
        """Should return empty list for empty string."""
        from app.models import WordVersion

        version = WordVersion(
            word_id=1,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            translations="",
        )
        assert version.translations_list == []

    def test_translations_list_none(self):
        """Should return empty list for None."""
        from app.models import WordVersion

        version = WordVersion(
            word_id=1,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            translations=None,
        )
        assert version.translations_list == []

    def test_translations_list_invalid_json(self):
        """Should return empty list for invalid JSON."""
        from app.models import WordVersion

        version = WordVersion(
            word_id=1,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            translations="not valid json",
        )
        assert version.translations_list == []


class TestWordEventTranslationsList:
    """Tests for WordEvent.translations_list property."""

    def test_translations_list_valid(self):
        """Should parse valid JSON array."""
        from app.models import WordEvent

        event = WordEvent(
            word_id=1,
            event_type="created",
            lemma="Test",
            pos="NOUN",
            translations='["a", "b"]',
        )
        assert event.translations_list == ["a", "b"]

    def test_translations_list_empty(self):
        """Should return empty list for empty."""
        from app.models import WordEvent

        event = WordEvent(
            word_id=1,
            event_type="created",
            lemma="Test",
            pos="NOUN",
            translations="",
        )
        assert event.translations_list == []

    def test_translations_list_none(self):
        """Should return empty list for None."""
        from app.models import WordEvent

        event = WordEvent(
            word_id=1,
            event_type="created",
            lemma="Test",
            pos="NOUN",
            translations=None,
        )
        assert event.translations_list == []

    def test_translations_list_invalid(self):
        """Should return empty list for invalid JSON."""
        from app.models import WordEvent

        event = WordEvent(
            word_id=1,
            event_type="created",
            lemma="Test",
            pos="NOUN",
            translations="[broken",
        )
        assert event.translations_list == []
