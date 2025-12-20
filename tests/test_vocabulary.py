"""Tests for vocabulary routes."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Word


class TestListVocabulary:
    """Tests for list_vocabulary route."""

    @pytest.mark.asyncio
    async def test_list_with_words(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should list vocabulary words."""
        word1 = Word(lemma="Arbeit", pos="NOUN", gender="die")
        word2 = Word(lemma="arbeiten", pos="VERB")
        async_session.add_all([word1, word2])
        await async_session.commit()

        response = await async_client.get("/vocabulary")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_with_search(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should filter by search term."""
        word1 = Word(lemma="Arbeit", pos="NOUN")
        word2 = Word(lemma="Buch", pos="NOUN")
        async_session.add_all([word1, word2])
        await async_session.commit()

        response = await async_client.get("/vocabulary?search=Arbeit")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_with_pos_filter(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should filter by POS."""
        word1 = Word(lemma="Arbeit", pos="NOUN")
        word2 = Word(lemma="schnell", pos="ADJ")
        async_session.add_all([word1, word2])
        await async_session.commit()

        response = await async_client.get("/vocabulary?pos=NOUN")
        assert response.status_code == 200


class TestEditWord:
    """Tests for word editing routes."""

    @pytest.mark.asyncio
    async def test_get_edit_form(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should return edit form for word."""
        word = Word(lemma="Arbeit", pos="NOUN", gender="die")
        async_session.add(word)
        await async_session.commit()

        response = await async_client.get(f"/vocabulary/{word.id}/edit")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_word_success(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should update word fields."""
        word = Word(lemma="Arbeit", pos="NOUN")
        async_session.add(word)
        await async_session.commit()

        response = await async_client.put(
            f"/vocabulary/{word.id}",
            data={
                "gender": "die",
                "plural": "Arbeiten",
                "translations": "work, job",
            },
        )
        assert response.status_code == 200

        await async_session.refresh(word)
        assert word.gender == "die"
        assert word.plural == "Arbeiten"

    @pytest.mark.asyncio
    async def test_update_verb_fields(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should update verb-specific fields."""
        word = Word(lemma="arbeiten", pos="VERB")
        async_session.add(word)
        await async_session.commit()

        response = await async_client.put(
            f"/vocabulary/{word.id}",
            data={
                "preterite": "arbeitete",
                "past_participle": "gearbeitet",
                "auxiliary": "haben",
                "translations": "to work",
            },
        )
        assert response.status_code == 200

        await async_session.refresh(word)
        assert word.preterite == "arbeitete"
        assert word.past_participle == "gearbeitet"

    @pytest.mark.asyncio
    async def test_update_clears_sync_status(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should clear Anki sync status when word is updated."""
        from datetime import datetime, timezone

        word = Word(
            lemma="Arbeit",
            pos="NOUN",
            anki_note_id=12345,
            anki_synced_at=datetime.now(timezone.utc),
        )
        async_session.add(word)
        await async_session.commit()

        response = await async_client.put(
            f"/vocabulary/{word.id}",
            data={"translations": "new translation"},
        )
        assert response.status_code == 200

        await async_session.refresh(word)
        assert word.anki_synced_at is None
