"""Tests for event sourcing service."""

import json

import pytest
from sqlalchemy import select

from app.models import Word, WordEvent
from app.services.events import (
    get_deleted_words,
    get_word_history,
    record_event,
    revert_to_event,
    undo_last_change,
)


class TestRecordEvent:
    """Tests for record_event function."""

    @pytest.mark.asyncio
    async def test_record_event_creates_event(self, async_session):
        """Should create a WordEvent with word snapshot."""
        word = Word(
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
            plural="Arbeiten",
            translations=json.dumps(["work"]),
        )
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        event = await record_event(async_session, word, "CREATED", "test", "Initial creation")
        await async_session.commit()

        assert event.id is not None
        assert event.word_id == word.id
        assert event.event_type == "CREATED"
        assert event.lemma == "Arbeit"
        assert event.pos == "NOUN"
        assert event.gender == "die"
        assert event.source == "test"
        assert event.reason == "Initial creation"

    @pytest.mark.asyncio
    async def test_record_event_without_reason(self, async_session):
        """Should create event with None reason."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        event = await record_event(async_session, word, "MODIFIED", "cli")
        await async_session.commit()

        assert event.reason is None


class TestGetWordHistory:
    """Tests for get_word_history function."""

    @pytest.mark.asyncio
    async def test_get_word_history_returns_events(self, async_session):
        """Should return events for a word ordered by date desc."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Create multiple events
        await record_event(async_session, word, "CREATED", "test", "First")
        word.lemma = "Updated"
        await record_event(async_session, word, "MODIFIED", "test", "Second")
        await async_session.commit()

        history = await get_word_history(async_session, word.id)

        assert len(history) == 2
        assert history[0].reason == "Second"  # Most recent first
        assert history[1].reason == "First"

    @pytest.mark.asyncio
    async def test_get_word_history_respects_limit(self, async_session):
        """Should limit number of events returned."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Create 5 events
        for i in range(5):
            await record_event(async_session, word, "MODIFIED", "test", f"Event {i}")
        await async_session.commit()

        history = await get_word_history(async_session, word.id, limit=3)

        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_word_history_empty(self, async_session):
        """Should return empty list for word with no events."""
        history = await get_word_history(async_session, 9999)
        assert history == []


class TestGetDeletedWords:
    """Tests for get_deleted_words function."""

    @pytest.mark.asyncio
    async def test_get_deleted_words_returns_deleted_events(self, async_session):
        """Should return only DELETED events."""
        word1 = Word(lemma="Deleted1", pos="NOUN")
        word2 = Word(lemma="Deleted2", pos="NOUN")
        word3 = Word(lemma="Active", pos="NOUN")
        async_session.add_all([word1, word2, word3])
        await async_session.commit()

        await record_event(async_session, word1, "DELETED", "test")
        await record_event(async_session, word2, "DELETED", "test")
        await record_event(async_session, word3, "CREATED", "test")
        await async_session.commit()

        deleted = await get_deleted_words(async_session)

        assert len(deleted) == 2
        assert all(e.event_type == "DELETED" for e in deleted)

    @pytest.mark.asyncio
    async def test_get_deleted_words_respects_limit(self, async_session):
        """Should limit number of deleted words returned."""
        for i in range(5):
            word = Word(lemma=f"Word{i}", pos="NOUN")
            async_session.add(word)
            await async_session.commit()
            await async_session.refresh(word)
            await record_event(async_session, word, "DELETED", "test")
        await async_session.commit()

        deleted = await get_deleted_words(async_session, limit=2)

        assert len(deleted) == 2

    @pytest.mark.asyncio
    async def test_get_deleted_words_with_after_filter(self, async_session):
        """Should filter deletions after a specific time."""
        from datetime import datetime, timezone

        word1 = Word(lemma="OldDelete", pos="NOUN")
        async_session.add(word1)
        await async_session.commit()
        await async_session.refresh(word1)
        await record_event(async_session, word1, "DELETED", "test")
        await async_session.commit()

        # Capture time between deletions
        cutoff = datetime.now(timezone.utc)

        word2 = Word(lemma="NewDelete", pos="NOUN")
        async_session.add(word2)
        await async_session.commit()
        await async_session.refresh(word2)
        await record_event(async_session, word2, "DELETED", "test")
        await async_session.commit()

        # Get only deletions after cutoff
        deleted = await get_deleted_words(async_session, after=cutoff)

        assert len(deleted) == 1
        assert deleted[0].lemma == "NewDelete"


class TestRevertToEvent:
    """Tests for revert_to_event function."""

    @pytest.mark.asyncio
    async def test_revert_updates_existing_word(self, async_session):
        """Should update existing word to match event state."""
        word = Word(lemma="Original", pos="NOUN", gender="die")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Record initial state
        event = await record_event(async_session, word, "CREATED", "test")
        await async_session.commit()

        # Modify word
        word.lemma = "Modified"
        word.gender = "der"
        await async_session.commit()

        # Revert to original
        restored = await revert_to_event(async_session, event)
        await async_session.commit()

        assert restored.lemma == "Original"
        assert restored.gender == "die"

    @pytest.mark.asyncio
    async def test_revert_recreates_deleted_word(self, async_session):
        """Should recreate word if it was deleted."""
        word = Word(lemma="ToDelete", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)
        word_id = word.id

        # Record state before deletion
        event = await record_event(async_session, word, "CREATED", "test")
        await async_session.commit()

        # Delete the word
        await async_session.delete(word)
        await async_session.commit()

        # Verify deleted
        deleted_word = await async_session.get(Word, word_id)
        assert deleted_word is None

        # Revert (recreate)
        restored = await revert_to_event(async_session, event)
        await async_session.commit()

        assert restored.id == word_id
        assert restored.lemma == "ToDelete"

    @pytest.mark.asyncio
    async def test_revert_records_restore_event(self, async_session):
        """Should record a RESTORED event after reverting."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        event = await record_event(async_session, word, "CREATED", "test")
        await async_session.commit()

        await revert_to_event(async_session, event, reason="Custom reason")
        await async_session.commit()

        # Check for RESTORED event
        stmt = select(WordEvent).where(
            WordEvent.word_id == word.id, WordEvent.event_type == "RESTORED"
        )
        result = await async_session.execute(stmt)
        restore_event = result.scalar_one()

        assert restore_event.source == "revert"
        assert restore_event.reason == "Custom reason"


class TestUndoLastChange:
    """Tests for undo_last_change function."""

    @pytest.mark.asyncio
    async def test_undo_reverts_to_previous_state(self, async_session):
        """Should revert word to previous event state."""
        word = Word(lemma="First", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Record first state
        await record_event(async_session, word, "CREATED", "test")
        await async_session.commit()

        # Modify and record second state
        word.lemma = "Second"
        await record_event(async_session, word, "MODIFIED", "test")
        await async_session.commit()

        # Undo
        restored = await undo_last_change(async_session, word.id)
        await async_session.commit()

        assert restored is not None
        assert restored.lemma == "First"

    @pytest.mark.asyncio
    async def test_undo_returns_none_with_no_history(self, async_session):
        """Should return None if no previous state exists."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Only one event
        await record_event(async_session, word, "CREATED", "test")
        await async_session.commit()

        result = await undo_last_change(async_session, word.id)

        assert result is None

    @pytest.mark.asyncio
    async def test_undo_returns_none_for_nonexistent_word(self, async_session):
        """Should return None for word with no events."""
        result = await undo_last_change(async_session, 9999)
        assert result is None
