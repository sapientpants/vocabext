"""SQLAlchemy ORM models."""

import json
from datetime import datetime
from typing import Any, Optional, cast

from sqlalchemy import ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Document(Base):
    """Uploaded document for vocabulary extraction."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text, unique=True)
    status: Mapped[str] = mapped_column(
        Text, default="processing"
    )  # processing, pending_review, reviewed, error
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    extractions: Mapped[list["Extraction"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    @property
    def pending_count(self) -> int:
        """Count of pending extractions."""
        return sum(1 for e in self.extractions if e.status == "pending")

    @property
    def duplicate_count(self) -> int:
        """Count of duplicate extractions."""
        return sum(1 for e in self.extractions if e.status == "duplicate")


class Word(Base):
    """Accepted vocabulary word in the user's collection."""

    __tablename__ = "words"
    __table_args__ = (UniqueConstraint("lemma", "pos", "gender", name="uq_word_identity"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    lemma: Mapped[str] = mapped_column(Text, index=True)
    pos: Mapped[str] = mapped_column(Text)  # NOUN, VERB, ADJ, ADV, ADP
    gender: Mapped[str | None] = mapped_column(Text, nullable=True)  # der/die/das (nouns only)
    plural: Mapped[str | None] = mapped_column(Text, nullable=True)  # nouns only
    preterite: Mapped[str | None] = mapped_column(Text, nullable=True)  # verbs only
    past_participle: Mapped[str | None] = mapped_column(Text, nullable=True)  # verbs only
    auxiliary: Mapped[str | None] = mapped_column(Text, nullable=True)  # haben/sein (verbs only)
    translations: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    anki_note_id: Mapped[int | None] = mapped_column(nullable=True)
    anki_synced_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    extractions: Mapped[list["Extraction"]] = relationship(back_populates="word")

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


class Extraction(Base):
    """Word extraction from a document, pending review."""

    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    word_id: Mapped[int | None] = mapped_column(ForeignKey("words.id"), nullable=True)
    surface_form: Mapped[str] = mapped_column(Text)
    lemma: Mapped[str] = mapped_column(Text)
    pos: Mapped[str] = mapped_column(Text)
    gender: Mapped[str | None] = mapped_column(Text, nullable=True)
    plural: Mapped[str | None] = mapped_column(Text, nullable=True)
    preterite: Mapped[str | None] = mapped_column(Text, nullable=True)
    past_participle: Mapped[str | None] = mapped_column(Text, nullable=True)
    auxiliary: Mapped[str | None] = mapped_column(Text, nullable=True)
    translations: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    context_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Text, default="pending"
    )  # pending, accepted, rejected, duplicate
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="extractions")
    word: Mapped[Optional["Word"]] = relationship(back_populates="extractions")

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
