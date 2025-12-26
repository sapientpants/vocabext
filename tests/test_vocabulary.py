"""Tests for vocabulary routes."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Word
from app.routes.vocabulary import build_diff
from app.services.enricher import EnrichmentResult


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
                "lemma": "Arbeit",
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
                "lemma": "arbeiten",
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
            data={"lemma": "Arbeit", "translations": "new translation"},
        )
        assert response.status_code == 200

        await async_session.refresh(word)
        assert word.anki_synced_at is None


class TestBuildDiff:
    """Tests for build_diff helper function."""

    def test_build_diff_noun_all_different(self):
        """Should detect all differences for a noun."""
        word = Word(lemma="Arbeit", pos="NOUN", gender="der", plural=None, translations=None)
        suggestions = EnrichmentResult(
            lemma="Arbeiten",
            gender="die",
            plural="Arbeiten",
            translations=["work", "job"],
        )
        diff = build_diff(word, suggestions)
        assert "lemma" in diff
        assert diff["lemma"]["current"] == "Arbeit"
        assert diff["lemma"]["suggested"] == "Arbeiten"
        assert "gender" in diff
        assert "plural" in diff
        assert "translations" in diff

    def test_build_diff_verb(self):
        """Should compare verb-specific fields."""
        word = Word(lemma="arbeiten", pos="VERB", preterite=None, past_participle=None)
        suggestions = EnrichmentResult(
            lemma="arbeiten",
            preterite="arbeitete",
            past_participle="gearbeitet",
            auxiliary="haben",
            translations=["to work"],
        )
        diff = build_diff(word, suggestions)
        assert "lemma" not in diff  # Same lemma
        assert "preterite" in diff
        assert "past_participle" in diff
        assert "auxiliary" in diff
        assert "translations" in diff

    def test_build_diff_no_differences(self):
        """Should return empty dict when no differences."""
        word = Word(lemma="schnell", pos="ADJ", translations='["fast", "quick"]')
        suggestions = EnrichmentResult(lemma="schnell", translations=["fast", "quick"])
        diff = build_diff(word, suggestions)
        assert diff == {}


class TestDeleteWord:
    """Tests for delete_word route."""

    @pytest.mark.asyncio
    async def test_delete_word_success(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should delete word and return empty response."""
        word = Word(lemma="Arbeit", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        word_id = word.id

        response = await async_client.delete(f"/vocabulary/{word_id}")
        assert response.status_code == 200
        assert response.text == ""

        # Verify word is deleted
        deleted = await async_session.get(Word, word_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_word(self, async_client: AsyncClient):
        """Should return 404 for nonexistent word."""
        response = await async_client.delete("/vocabulary/99999")
        assert response.status_code == 404


class TestValidateWithLLM:
    """Tests for validate_with_llm route."""

    @pytest.mark.asyncio
    async def test_validate_success(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should return diff view with suggestions."""
        word = Word(lemma="Arbeit", pos="NOUN", gender="die")
        async_session.add(word)
        await async_session.commit()

        with patch("app.routes.vocabulary.Enricher") as MockEnricher:
            mock_instance = MockEnricher.return_value
            mock_instance.validate_lemma = AsyncMock(
                return_value={"valid": True, "corrected_lemma": "Arbeit", "error": None}
            )
            mock_instance.enrich = AsyncMock(
                return_value=EnrichmentResult(
                    lemma="Arbeit",
                    gender="die",
                    plural="Arbeiten",
                    translations=["work", "job"],
                )
            )

            response = await async_client.post(
                f"/vocabulary/{word.id}/validate",
                data={"context_sentence": "Die Arbeit ist wichtig."},
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_with_error(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should display error when LLM fails."""
        word = Word(lemma="Arbeit", pos="NOUN")
        async_session.add(word)
        await async_session.commit()

        with patch("app.routes.vocabulary.Enricher") as MockEnricher:
            mock_instance = MockEnricher.return_value
            mock_instance.validate_lemma = AsyncMock(
                return_value={
                    "valid": True,
                    "corrected_lemma": "Arbeit",
                    "error": "Connection failed",
                }
            )

            response = await async_client.post(f"/vocabulary/{word.id}/validate", data={})
            assert response.status_code == 200
            assert b"Connection failed" in response.content


class TestApplySuggestions:
    """Tests for apply_suggestions route."""

    @pytest.mark.asyncio
    async def test_apply_selected_suggestions(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should apply only selected suggestions."""
        word = Word(lemma="Arbeit", pos="NOUN", gender="der")
        async_session.add(word)
        await async_session.commit()

        response = await async_client.post(
            f"/vocabulary/{word.id}/apply-suggestions",
            data={
                "apply_gender": "1",
                "suggested_gender": "die",
                "apply_plural": "1",
                "suggested_plural": "Arbeiten",
                # translations not selected
            },
        )
        assert response.status_code == 200

        await async_session.refresh(word)
        assert word.gender == "die"
        assert word.plural == "Arbeiten"

    @pytest.mark.asyncio
    async def test_apply_translations(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should apply translation suggestions."""
        word = Word(lemma="Arbeit", pos="NOUN")
        async_session.add(word)
        await async_session.commit()

        response = await async_client.post(
            f"/vocabulary/{word.id}/apply-suggestions",
            data={
                "apply_translations": "1",
                "suggested_translations": "work,job,labor",
            },
        )
        assert response.status_code == 200

        await async_session.refresh(word)
        assert word.translations_list == ["work", "job", "labor"]

    @pytest.mark.asyncio
    async def test_apply_clears_sync_status(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should clear Anki sync status when applying suggestions."""
        from datetime import datetime, timezone

        word = Word(
            lemma="Arbeit",
            pos="NOUN",
            anki_note_id=12345,
            anki_synced_at=datetime.now(timezone.utc),
        )
        async_session.add(word)
        await async_session.commit()

        response = await async_client.post(
            f"/vocabulary/{word.id}/apply-suggestions",
            data={"apply_gender": "1", "suggested_gender": "die"},
        )
        assert response.status_code == 200

        await async_session.refresh(word)
        assert word.anki_synced_at is None
