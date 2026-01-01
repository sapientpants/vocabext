"""Event sourcing service for tracking word changes."""

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Word, WordEvent

logger = logging.getLogger(__name__)


async def record_event(
    session: AsyncSession,
    word: Word,
    event_type: str,
    source: str,
    reason: str | None = None,
) -> WordEvent:
    """
    Record an event with the current word state as a snapshot.

    Args:
        session: Database session
        word: The word to snapshot
        event_type: One of CREATED, MODIFIED, DELETED, RESTORED
        source: Where the event originated (validate, edit, import, cli)
        reason: Optional explanation for the change

    Returns:
        The created WordEvent
    """
    event = WordEvent(
        word_id=word.id,
        event_type=event_type,
        lemma=word.lemma,
        pos=word.pos,
        gender=word.gender,
        plural=word.plural,
        preterite=word.preterite,
        past_participle=word.past_participle,
        auxiliary=word.auxiliary,
        translations=word.translations,
        definition_de=word.definition_de,
        synonyms=word.synonyms,
        frequency=word.frequency,
        ipa=word.ipa,
        lemma_source=word.lemma_source,
        dictionary_url=word.dictionary_url,
        anki_note_id=word.anki_note_id,
        source=source,
        reason=reason,
    )
    session.add(event)
    await session.flush()

    logger.debug(
        "Recorded %s event for word %d (%s): %s",
        event_type,
        word.id,
        word.lemma,
        reason or "-",
    )

    return event


async def get_word_history(
    session: AsyncSession,
    word_id: int,
    limit: int = 50,
) -> list[WordEvent]:
    """
    Get the event history for a word.

    Args:
        session: Database session
        word_id: The word ID to get history for
        limit: Maximum number of events to return

    Returns:
        List of WordEvent, most recent first
    """
    stmt = (
        select(WordEvent)
        .where(WordEvent.word_id == word_id)
        .order_by(WordEvent.event_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_deleted_words(
    session: AsyncSession,
    after: datetime | None = None,
    limit: int = 100,
) -> list[WordEvent]:
    """
    Get the most recent DELETED events (for words that can be restored).

    Args:
        session: Database session
        after: Only get deletions after this time
        limit: Maximum number to return

    Returns:
        List of DELETED WordEvents, most recent first
    """
    stmt = (
        select(WordEvent)
        .where(WordEvent.event_type == "DELETED")
        .order_by(WordEvent.event_at.desc())
        .limit(limit)
    )

    if after:
        stmt = stmt.where(WordEvent.event_at > after)

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def revert_to_event(
    session: AsyncSession,
    event: WordEvent,
    source: str = "revert",
    reason: str | None = None,
) -> Word:
    """
    Restore a word to the state captured in an event.

    If the word was deleted, this recreates it.
    If the word exists, this updates it to match the event state.

    Args:
        session: Database session
        event: The event to restore state from
        source: Source to record for the restore event
        reason: Optional reason for the restore

    Returns:
        The restored/updated Word
    """
    # Check if word still exists
    word: Word | None = await session.get(Word, event.word_id)

    if word is None:
        # Word was deleted - recreate it
        word = Word(
            id=event.word_id,
            lemma=event.lemma,
            pos=event.pos,
            gender=event.gender,
            plural=event.plural,
            preterite=event.preterite,
            past_participle=event.past_participle,
            auxiliary=event.auxiliary,
            translations=event.translations,
            definition_de=event.definition_de,
            synonyms=event.synonyms,
            frequency=event.frequency,
            ipa=event.ipa,
            lemma_source=event.lemma_source,
            dictionary_url=event.dictionary_url,
            anki_note_id=event.anki_note_id,
        )
        session.add(word)
        await session.flush()

        logger.info("Restored deleted word %d (%s)", word.id, word.lemma)
    else:
        # Word exists - update it to match event state
        word.lemma = event.lemma
        word.pos = event.pos
        word.gender = event.gender
        word.plural = event.plural
        word.preterite = event.preterite
        word.past_participle = event.past_participle
        word.auxiliary = event.auxiliary
        word.translations = event.translations
        word.definition_de = event.definition_de
        word.synonyms = event.synonyms
        word.frequency = event.frequency
        word.ipa = event.ipa
        word.lemma_source = event.lemma_source
        word.dictionary_url = event.dictionary_url
        # Don't restore anki_note_id - that's managed by sync

        logger.info("Reverted word %d (%s) to event %d", word.id, word.lemma, event.id)

    # Record the restore as a new event
    restore_reason = reason or f"Reverted to event {event.id}"
    await record_event(session, word, "RESTORED", source, restore_reason)

    return word


async def undo_last_change(
    session: AsyncSession,
    word_id: int,
) -> Word | None:
    """
    Undo the last change to a word by reverting to the previous state.

    Args:
        session: Database session
        word_id: The word ID to undo

    Returns:
        The restored Word, or None if no previous state exists
    """
    # Get the last two events for this word
    stmt = (
        select(WordEvent)
        .where(WordEvent.word_id == word_id)
        .order_by(WordEvent.event_at.desc())
        .limit(2)
    )
    result = await session.execute(stmt)
    events = list(result.scalars().all())

    if len(events) < 2:
        logger.warning("Cannot undo - no previous state for word %d", word_id)
        return None

    # Revert to the second-to-last event (the state before the last change)
    previous_event = events[1]
    return await revert_to_event(session, previous_event, source="undo", reason="Undo last change")
