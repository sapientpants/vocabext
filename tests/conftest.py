"""Pytest configuration and fixtures."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_session
from app.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def test_app(async_session: AsyncSession) -> FastAPI:
    """Create a test FastAPI application."""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield async_session

    app.dependency_overrides[get_session] = override_get_session
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    """Create a synchronous test client."""
    return TestClient(test_app)


@pytest.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an asynchronous test client."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def sample_word_data() -> dict[str, Any]:
    """Sample word data for testing."""
    return {
        "lemma": "Arbeit",
        "pos": "NOUN",
        "gender": "die",
        "plural": "Arbeiten",
        "translations": '["work", "job", "labor"]',
    }


@pytest.fixture
def sample_extraction_data() -> dict[str, Any]:
    """Sample extraction data for testing."""
    return {
        "surface_form": "Arbeit",
        "lemma": "Arbeit",
        "pos": "NOUN",
        "gender": "die",
        "plural": "Arbeiten",
        "translations": '["work", "job"]',
        "context_sentence": "Die Arbeit ist wichtig.",
        "status": "pending",
    }


@pytest.fixture
def temp_upload_dir(tmp_path: Path) -> Path:
    """Create a temporary upload directory."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir
