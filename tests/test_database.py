"""Tests for database configuration."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base


class TestDatabase:
    """Tests for database module."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self, async_engine):
        """Should create all tables."""
        # Tables are already created in the fixture, verify they exist
        async with async_engine.connect() as conn:
            # Check that tables exist by querying sqlite_master
            result = await conn.execute(
                __import__("sqlalchemy").text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = [row[0] for row in result.fetchall()]

        assert "words" in tables

    @pytest.mark.asyncio
    async def test_session_yields_async_session(self, async_session):
        """Should yield an AsyncSession."""
        assert isinstance(async_session, AsyncSession)

    @pytest.mark.asyncio
    async def test_session_can_execute_queries(self, async_session):
        """Should be able to execute queries."""
        from app.models import Word

        # Create a word
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()

        # Query it back
        result = await async_session.get(Word, word.id)
        assert result is not None
        assert result.lemma == "Test"


class TestBase:
    """Tests for Base class."""

    def test_base_is_declarative(self):
        """Should be a valid SQLAlchemy DeclarativeBase."""
        from sqlalchemy.orm import DeclarativeBase

        assert issubclass(Base, DeclarativeBase)
