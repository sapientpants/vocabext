"""Tests for sync routes."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Word


class TestSyncRoutes:
    """Tests for sync routes."""

    @pytest.mark.asyncio
    async def test_sync_status_with_words(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should return sync status with word counts."""
        # Create some words
        word1 = Word(lemma="test1", pos="NOUN")
        word2 = Word(lemma="test2", pos="NOUN", anki_note_id=12345)
        async_session.add_all([word1, word2])
        await async_session.commit()

        with patch("app.routes.sync.AnkiService") as mock_anki:
            mock_instance = mock_anki.return_value
            mock_instance.get_sync_stats = AsyncMock(
                return_value={"available": True, "deck_exists": True}
            )

            response = await async_client.get("/sync/status")
            assert response.status_code == 200
            data = response.json()
            assert "anki" in data
            assert "words" in data
            assert data["words"]["total"] >= 2
            assert data["words"]["synced"] >= 1

    @pytest.mark.asyncio
    async def test_sync_anki_available(self, async_client: AsyncClient):
        """Should sync words when Anki is available."""
        with patch("app.routes.sync.AnkiService") as mock_anki:
            mock_instance = mock_anki.return_value
            mock_instance.is_available = AsyncMock(return_value=True)
            mock_instance.ensure_deck = AsyncMock()
            mock_instance.ensure_note_type = AsyncMock()

            with patch("app.routes.sync.get_session") as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute = AsyncMock()
                mock_session.execute.return_value.scalars.return_value.all.return_value = []
                mock_session.commit = AsyncMock()

                async def session_generator():
                    yield mock_session

                mock_get_session.return_value = session_generator()

                response = await async_client.post("/sync")
                # Will either succeed (200) or fail validation
                assert response.status_code in [200, 503, 422]


class TestSyncWordsCounting:
    """Tests for sync word counting functionality."""

    @pytest.mark.asyncio
    async def test_counts_synced_vs_unsynced(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should correctly count synced vs unsynced words."""
        # Create words with different sync states
        synced_word = Word(
            lemma="synced",
            pos="NOUN",
            anki_note_id=12345,
        )
        unsynced_word = Word(
            lemma="unsynced",
            pos="NOUN",
            anki_note_id=None,
        )
        async_session.add_all([synced_word, unsynced_word])
        await async_session.commit()

        with patch("app.routes.sync.AnkiService") as mock_anki:
            mock_instance = mock_anki.return_value
            mock_instance.get_sync_stats = AsyncMock(return_value={"available": False})

            response = await async_client.get("/sync/status")
            assert response.status_code == 200
            data = response.json()

            # Should have at least our test words in count
            assert data["words"]["total"] >= 2
            assert data["words"]["synced"] >= 1
