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

        assert "documents" in tables
        assert "words" in tables
        assert "extractions" in tables

    @pytest.mark.asyncio
    async def test_session_yields_async_session(self, async_session):
        """Should yield an AsyncSession."""
        assert isinstance(async_session, AsyncSession)

    @pytest.mark.asyncio
    async def test_session_can_execute_queries(self, async_session):
        """Should be able to execute queries."""
        from app.models import Document

        # Create a document
        doc = Document(filename="test.pdf", content_hash="testhash")
        async_session.add(doc)
        await async_session.commit()

        # Query it back
        result = await async_session.get(Document, doc.id)
        assert result is not None
        assert result.filename == "test.pdf"


class TestBase:
    """Tests for Base class."""

    def test_base_is_declarative(self):
        """Should be a valid SQLAlchemy DeclarativeBase."""
        from sqlalchemy.orm import DeclarativeBase

        assert issubclass(Base, DeclarativeBase)
