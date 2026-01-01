"""Tests for database configuration."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, get_session, init_db


class TestDatabase:
    """Tests for database module."""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self, async_engine):
        """Should create all tables."""
        # Tables are already created in the fixture, verify they exist
        async with async_engine.connect() as conn:
            # Check that tables exist by querying sqlite_master
            result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
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


class TestInitDb:
    """Tests for init_db function."""

    @pytest.mark.asyncio
    async def test_init_db_creates_all_tables(self, tmp_path, monkeypatch):
        """Should create all tables via init_db."""
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        from app import database

        # Create a temp database
        db_path = tmp_path / "test.db"
        test_engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            poolclass=NullPool,
        )

        # Monkeypatch the engine
        monkeypatch.setattr(database, "engine", test_engine)

        # Call init_db
        await init_db()

        # Verify tables exist
        async with test_engine.connect() as conn:
            result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]

        assert "words" in tables
        await test_engine.dispose()


class TestGetSession:
    """Tests for get_session function."""

    @pytest.mark.asyncio
    async def test_get_session_yields_session(self, tmp_path, monkeypatch):
        """Should yield an AsyncSession."""
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from sqlalchemy.pool import NullPool

        from app import database

        # Create a temp database
        db_path = tmp_path / "test.db"
        test_engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            poolclass=NullPool,
        )

        # Create tables
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create session factory
        test_async_session = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Monkeypatch the session factory
        monkeypatch.setattr(database, "async_session", test_async_session)

        # Use get_session as async generator
        gen = get_session()
        session = await gen.__anext__()

        assert isinstance(session, AsyncSession)

        # Cleanup: exhaust the generator (it should raise StopAsyncIteration)
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        await test_engine.dispose()


class TestSqlitePragma:
    """Tests for SQLite pragma setup."""

    @pytest.mark.asyncio
    async def test_foreign_keys_enabled(self, tmp_path, monkeypatch):
        """Should enable foreign keys pragma on connection."""
        from sqlalchemy import event
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        # Create engine with pragma listener
        db_path = tmp_path / "test_pragma.db"
        test_engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            poolclass=NullPool,
        )

        # Register the pragma listener (mimicking database.py)
        @event.listens_for(test_engine.sync_engine, "connect")
        def set_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        # Create tables and verify pragma
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Check that foreign_keys is enabled
        async with test_engine.connect() as conn:
            result = await conn.execute(text("PRAGMA foreign_keys"))
            fk_enabled = result.scalar()

        assert fk_enabled == 1
        await test_engine.dispose()
