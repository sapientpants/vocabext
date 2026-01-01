"""SQLAlchemy ORM models."""

import json
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy import ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utc_now() -> datetime:
    """Return current UTC time (replaces deprecated datetime.utcnow())."""
    return datetime.now(timezone.utc)


class Word(Base):
    """Accepted vocabulary word in the user's collection."""

    __tablename__ = "words"
    __table_args__ = (
        UniqueConstraint("lemma", "pos", "gender", name="uq_word_identity"),
        # Composite index for word lookups by lemma and pos
        Index("ix_words_lemma_pos", "lemma", "pos"),
        # Index for sync status queries
        Index("ix_words_anki_note_id", "anki_note_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    lemma: Mapped[str] = mapped_column(Text, index=True)
    pos: Mapped[str] = mapped_column(Text, index=True)  # NOUN, VERB, ADJ, ADV, ADP
    gender: Mapped[str | None] = mapped_column(Text, nullable=True)  # der/die/das (nouns only)
    plural: Mapped[str | None] = mapped_column(Text, nullable=True)  # nouns only
    preterite: Mapped[str | None] = mapped_column(Text, nullable=True)  # verbs only
    past_participle: Mapped[str | None] = mapped_column(Text, nullable=True)  # verbs only
    auxiliary: Mapped[str | None] = mapped_column(Text, nullable=True)  # haben/sein (verbs only)
    translations: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array

    # Dictionary-grounded fields
    definition_de: Mapped[str | None] = mapped_column(Text, nullable=True)  # German definition
    synonyms: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    frequency: Mapped[float | None] = mapped_column(nullable=True)  # DWDS 0-6 scale
    ipa: Mapped[str | None] = mapped_column(Text, nullable=True)  # Pronunciation
    lemma_source: Mapped[str | None] = mapped_column(Text, nullable=True)  # "spacy", "dwds", etc.
    dictionary_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # Link to entry

    anki_note_id: Mapped[int | None] = mapped_column(nullable=True)
    anki_synced_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utc_now)

    @property
    def translations_list(self) -> list[str]:
        """Get translations as a Python list."""
        if not self.translations:
            return []
        try:
            result: Any = json.loads(self.translations)
            return cast(list[str], result)
        except json.JSONDecodeError:
            return []

    @translations_list.setter
    def translations_list(self, value: list[str]) -> None:
        """Set translations from a Python list."""
        self.translations = json.dumps(value)

    @property
    def display_word(self) -> str:
        """Get word with article for nouns."""
        if self.pos == "NOUN" and self.gender:
            return f"{self.gender} {self.lemma}"
        return str(self.lemma)

    @property
    def grammar_info(self) -> str:
        """Get formatted grammar information."""
        parts: list[str] = []
        if self.plural:
            parts.append(f"pl: {self.plural}")
        if self.preterite:
            parts.append(f"prät: {self.preterite}")
        if self.past_participle:
            aux = self.auxiliary or "haben"
            parts.append(f"pp: {aux} {self.past_participle}")
        return ", ".join(parts)

    @property
    def is_synced(self) -> bool:
        """Check if word is synced to Anki."""
        return self.anki_note_id is not None

    # Versioning fields and relationships (added after initial definition)
    current_version: Mapped[int] = mapped_column(default=1)
    updated_at: Mapped[datetime] = mapped_column(default=_utc_now, onupdate=_utc_now)

    # Review flags for batch validation
    needs_review: Mapped[bool] = mapped_column(default=False)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    versions: Mapped[list["WordVersion"]] = relationship(
        back_populates="word",
        cascade="all, delete-orphan",
        order_by="WordVersion.version_number.desc()",
    )

    @property
    def needs_sync(self) -> bool:
        """Check if word has been modified since last sync."""
        if self.anki_synced_at is None:
            return True
        # If synced and no versions, no changes to sync
        if not self.versions:
            return False
        # Check if latest version was created after last sync
        latest = self.versions[0]
        return bool(latest.created_at > self.anki_synced_at)


class WordVersion(Base):
    """Historical version of a vocabulary word."""

    __tablename__ = "word_versions"
    __table_args__ = (Index("ix_word_versions_word_id_version", "word_id", "version_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id", ondelete="CASCADE"), index=True)
    version_number: Mapped[int] = mapped_column()

    # Snapshot of word data at this version
    lemma: Mapped[str] = mapped_column(Text)
    pos: Mapped[str] = mapped_column(Text)
    gender: Mapped[str | None] = mapped_column(Text, nullable=True)
    plural: Mapped[str | None] = mapped_column(Text, nullable=True)
    preterite: Mapped[str | None] = mapped_column(Text, nullable=True)
    past_participle: Mapped[str | None] = mapped_column(Text, nullable=True)
    auxiliary: Mapped[str | None] = mapped_column(Text, nullable=True)
    translations: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Dictionary-grounded fields (snapshot)
    definition_de: Mapped[str | None] = mapped_column(Text, nullable=True)
    synonyms: Mapped[str | None] = mapped_column(Text, nullable=True)
    frequency: Mapped[float | None] = mapped_column(nullable=True)
    ipa: Mapped[str | None] = mapped_column(Text, nullable=True)
    lemma_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    dictionary_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Version metadata
    created_at: Mapped[datetime] = mapped_column(default=_utc_now)

    # Relationship
    word: Mapped["Word"] = relationship(back_populates="versions")

    @property
    def translations_list(self) -> list[str]:
        """Get translations as a Python list."""
        if not self.translations:
            return []
        try:
            result: Any = json.loads(self.translations)
            return cast(list[str], result)
        except json.JSONDecodeError:
            return []


class WordEvent(Base):
    """
    Immutable event log for word operations (event sourcing).

    Events are retained indefinitely to support:
    - Full audit trail of vocabulary changes
    - Restoring deleted words via revert_to_event()
    - Debugging and analytics

    Retention Policy:
    - Events are never automatically deleted
    - Manual cleanup can be done via SQL if storage becomes a concern
    - Example: DELETE FROM word_events WHERE event_at < datetime('now', '-1 year')
    """

    __tablename__ = "word_events"
    __table_args__ = (
        Index("ix_word_events_word_id", "word_id"),
        Index("ix_word_events_event_at", "event_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # No FK constraint - word may be deleted but events are retained for audit trail
    # Index is defined in __table_args__ with explicit name
    word_id: Mapped[int] = mapped_column()
    event_type: Mapped[str] = mapped_column(Text)  # CREATED, MODIFIED, DELETED, RESTORED
    event_at: Mapped[datetime] = mapped_column(default=_utc_now)

    # Full snapshot of word state at this event
    lemma: Mapped[str] = mapped_column(Text)
    pos: Mapped[str] = mapped_column(Text)
    gender: Mapped[str | None] = mapped_column(Text, nullable=True)
    plural: Mapped[str | None] = mapped_column(Text, nullable=True)
    preterite: Mapped[str | None] = mapped_column(Text, nullable=True)
    past_participle: Mapped[str | None] = mapped_column(Text, nullable=True)
    auxiliary: Mapped[str | None] = mapped_column(Text, nullable=True)
    translations: Mapped[str | None] = mapped_column(Text, nullable=True)
    definition_de: Mapped[str | None] = mapped_column(Text, nullable=True)
    synonyms: Mapped[str | None] = mapped_column(Text, nullable=True)
    frequency: Mapped[float | None] = mapped_column(nullable=True)
    ipa: Mapped[str | None] = mapped_column(Text, nullable=True)
    lemma_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    dictionary_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Anki sync state at this event
    anki_note_id: Mapped[int | None] = mapped_column(nullable=True)

    # Event metadata
    source: Mapped[str] = mapped_column(Text)  # "validate", "edit", "import", "cli"
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    @property
    def translations_list(self) -> list[str]:
        """Get translations as a Python list."""
        if not self.translations:
            return []
        try:
            result: Any = json.loads(self.translations)
            return cast(list[str], result)
        except json.JSONDecodeError:
            return []
