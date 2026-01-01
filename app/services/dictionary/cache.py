"""Dictionary lookup cache using SQLite."""

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import Index, Text, UniqueConstraint, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.services.dictionary.base import DictionaryEntry

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


class DictionaryCache(Base):
    """Cache for dictionary lookups."""

    __tablename__ = "dictionary_cache"
    __table_args__ = (
        UniqueConstraint("word", "pos", "source", name="uq_cache_lookup"),
        Index("ix_cache_word_source", "word", "source"),
        Index("ix_cache_expires_at", "expires_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    word: Mapped[str] = mapped_column(Text, index=True)
    pos: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text)  # "dwds", "openthesaurus"
    data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON or null for "not found"
    created_at: Mapped[datetime] = mapped_column(default=_utc_now)
    expires_at: Mapped[datetime] = mapped_column()


def _serialize_entry(entry: DictionaryEntry) -> str:
    """Serialize a DictionaryEntry to JSON."""
    return json.dumps(
        {
            "lemma": entry.lemma,
            "lemma_validated": entry.lemma_validated,
            "pos": entry.pos,
            "gender": entry.gender,
            "definitions": entry.definitions,
            "synonyms": entry.synonyms,
            "frequency": entry.frequency,
            "ipa": entry.ipa,
            "source": entry.source,
            "url": entry.url,
        }
    )


def _deserialize_entry(data: str) -> DictionaryEntry:
    """Deserialize JSON to a DictionaryEntry."""
    obj = json.loads(data)
    return DictionaryEntry(
        lemma=obj["lemma"],
        lemma_validated=obj.get("lemma_validated", False),
        pos=obj.get("pos"),
        gender=obj.get("gender"),
        definitions=obj.get("definitions", []),
        synonyms=obj.get("synonyms", []),
        frequency=obj.get("frequency"),
        ipa=obj.get("ipa"),
        source=obj.get("source", ""),
        url=obj.get("url"),
    )


class CacheManager:
    """Manages dictionary cache operations."""

    DEFAULT_TTL_DAYS = 30
    NOT_FOUND_TTL_DAYS = 7

    def __init__(
        self,
        ttl_days: int = DEFAULT_TTL_DAYS,
        not_found_ttl_days: int = NOT_FOUND_TTL_DAYS,
    ) -> None:
        self.ttl_days = ttl_days
        self.not_found_ttl_days = not_found_ttl_days

    async def get(
        self,
        session: AsyncSession,
        word: str,
        pos: str | None,
        source: str,
    ) -> tuple[bool, DictionaryEntry | None]:
        """
        Get a cached entry.

        Returns:
            Tuple of (cache_hit, entry)
            - cache_hit: True if entry was found in cache (even if None/not-found)
            - entry: The cached DictionaryEntry, or None if not found or cache miss
        """
        now = _utc_now()
        stmt = select(DictionaryCache).where(
            DictionaryCache.word == word.lower(),
            DictionaryCache.pos == pos,
            DictionaryCache.source == source,
            DictionaryCache.expires_at > now,
        )
        result = await session.execute(stmt)
        cached = result.scalar_one_or_none()

        if cached is None:
            return False, None

        if cached.data is None:
            # Cached "not found" result
            return True, None

        return True, _deserialize_entry(cached.data)

    async def set(
        self,
        session: AsyncSession,
        word: str,
        pos: str | None,
        source: str,
        entry: DictionaryEntry | None,
    ) -> None:
        """
        Cache a dictionary entry (or "not found" if entry is None).
        """
        now = _utc_now()
        ttl = self.ttl_days if entry else self.not_found_ttl_days
        expires_at = now + timedelta(days=ttl)

        # Check if entry already exists
        stmt = select(DictionaryCache).where(
            DictionaryCache.word == word.lower(),
            DictionaryCache.pos == pos,
            DictionaryCache.source == source,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing
            existing.data = _serialize_entry(entry) if entry else None
            existing.created_at = now
            existing.expires_at = expires_at
        else:
            # Create new
            cache_entry = DictionaryCache(
                word=word.lower(),
                pos=pos,
                source=source,
                data=_serialize_entry(entry) if entry else None,
                created_at=now,
                expires_at=expires_at,
            )
            session.add(cache_entry)

        await session.flush()

    async def cleanup_expired(self, session: AsyncSession) -> int:
        """
        Remove expired cache entries.

        Returns the number of entries deleted.
        """
        from sqlalchemy import delete, func, select

        now = _utc_now()

        # Count before deleting
        count_stmt = (
            select(func.count())
            .select_from(DictionaryCache)
            .where(DictionaryCache.expires_at <= now)
        )
        count_result = await session.execute(count_stmt)
        count = count_result.scalar() or 0

        # Delete expired entries
        stmt = delete(DictionaryCache).where(DictionaryCache.expires_at <= now)
        await session.execute(stmt)
        await session.flush()

        return int(count)
