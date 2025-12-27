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
            follow_redirects=False,
        )
        # Now returns 303 redirect to detail page (versioning feature)
        assert response.status_code == 303

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
            follow_redirects=False,
        )
        # Now returns 303 redirect to detail page (versioning feature)
        assert response.status_code == 303

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
            follow_redirects=False,
        )
        # Now returns 303 redirect to detail page (versioning feature)
        assert response.status_code == 303

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
            follow_redirects=False,
        )
        # Now returns 303 redirect to detail page (versioning feature)
        assert response.status_code == 303

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
            follow_redirects=False,
        )
        # Now returns 303 redirect to detail page (versioning feature)
        assert response.status_code == 303

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
            follow_redirects=False,
        )
        # Now returns 303 redirect to detail page (versioning feature)
        assert response.status_code == 303

        await async_session.refresh(word)
        assert word.anki_synced_at is None


class TestVocabularyFilters:
    """Tests for vocabulary list filters."""

    @pytest.mark.asyncio
    async def test_filter_synced(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should filter by sync status."""
        from datetime import datetime, timezone

        synced = Word(
            lemma="Synced",
            pos="NOUN",
            anki_note_id=123,
            anki_synced_at=datetime.now(timezone.utc),
        )
        unsynced = Word(lemma="Unsynced", pos="NOUN")
        async_session.add_all([synced, unsynced])
        await async_session.commit()

        response = await async_client.get("/vocabulary?sync_status=synced")
        assert response.status_code == 200
        assert "Synced" in response.text
        # Unsynced word should not be in filtered result
        assert response.text.count("word-row") == 1

    @pytest.mark.asyncio
    async def test_filter_unsynced(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should filter by unsynced status."""
        from datetime import datetime, timezone

        synced = Word(
            lemma="Synced",
            pos="NOUN",
            anki_note_id=123,
            anki_synced_at=datetime.now(timezone.utc),
        )
        unsynced = Word(lemma="Unsynced", pos="NOUN")
        async_session.add_all([synced, unsynced])
        await async_session.commit()

        response = await async_client.get("/vocabulary?sync_status=unsynced")
        assert response.status_code == 200
        assert "Unsynced" in response.text

    @pytest.mark.asyncio
    async def test_filter_version_v1(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should filter by version v1."""
        v1_word = Word(lemma="V1Word", pos="NOUN", current_version=1)
        v2_word = Word(lemma="V2Word", pos="NOUN", current_version=2)
        async_session.add_all([v1_word, v2_word])
        await async_session.commit()

        response = await async_client.get("/vocabulary?version=v1")
        assert response.status_code == 200
        assert "V1Word" in response.text

    @pytest.mark.asyncio
    async def test_filter_version_v2plus(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should filter by version v2+."""
        v1_word = Word(lemma="V1Word", pos="NOUN", current_version=1)
        v2_word = Word(lemma="V2Word", pos="NOUN", current_version=2)
        async_session.add_all([v1_word, v2_word])
        await async_session.commit()

        response = await async_client.get("/vocabulary?version=v2%2B")
        assert response.status_code == 200
        assert "V2Word" in response.text

    @pytest.mark.asyncio
    async def test_filter_needs_review(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should filter by review status."""
        needs_review = Word(
            lemma="NeedsReview", pos="NOUN", needs_review=True, review_reason="Test"
        )
        reviewed = Word(lemma="Reviewed", pos="NOUN", needs_review=False)
        async_session.add_all([needs_review, reviewed])
        await async_session.commit()

        response = await async_client.get("/vocabulary?review_status=needs_review")
        assert response.status_code == 200
        assert "NeedsReview" in response.text

    @pytest.mark.asyncio
    async def test_filter_reviewed(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should filter by reviewed status."""
        needs_review = Word(
            lemma="NeedsReview", pos="NOUN", needs_review=True, review_reason="Test"
        )
        reviewed = Word(lemma="Reviewed", pos="NOUN", needs_review=False)
        async_session.add_all([needs_review, reviewed])
        await async_session.commit()

        response = await async_client.get("/vocabulary?review_status=reviewed")
        assert response.status_code == 200
        assert "Reviewed" in response.text

    @pytest.mark.asyncio
    async def test_filter_updated_within(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should filter by updated_within."""
        recent = Word(
            lemma="Recent",
            pos="NOUN",
        )
        async_session.add(recent)
        await async_session.commit()

        response = await async_client.get("/vocabulary?updated_within=1")
        assert response.status_code == 200
        assert "Recent" in response.text

    @pytest.mark.asyncio
    async def test_filter_needs_resync(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should filter by needs_resync status."""
        from datetime import datetime, timezone

        # Synced word (has note_id AND synced_at)
        synced = Word(
            lemma="Synced",
            pos="NOUN",
            anki_note_id=123,
            anki_synced_at=datetime.now(timezone.utc),
        )
        # Needs resync (has note_id but no synced_at - was modified)
        needs_resync = Word(
            lemma="NeedsResync",
            pos="NOUN",
            anki_note_id=456,
            anki_synced_at=None,
        )
        async_session.add_all([synced, needs_resync])
        await async_session.commit()

        response = await async_client.get("/vocabulary?sync_status=needs_resync")
        assert response.status_code == 200
        assert "NeedsResync" in response.text
        # Synced word should not be in needs_resync results
        assert response.text.count("word-row") == 1


class TestBuildFilteredQuery:
    """Tests for build_filtered_query helper."""

    def test_default_order(self):
        """Should order by lemma by default."""
        from app.routes.vocabulary import build_filtered_query

        stmt = build_filtered_query()
        # The query should have order_by
        assert str(stmt).find("ORDER BY") != -1

    def test_random_order(self):
        """Should order randomly when specified."""
        from app.routes.vocabulary import build_filtered_query

        stmt = build_filtered_query(random_order=True)
        # The query should have random() in ORDER BY
        query_str = str(stmt)
        assert "ORDER BY" in query_str


class TestBatchValidationHelpers:
    """Tests for batch validation helper functions."""

    @pytest.mark.asyncio
    async def test_apply_enrichment_modified(self, async_session: AsyncSession):
        """Should return 'modified' when word is updated."""
        from app.routes.vocabulary import apply_enrichment_to_word

        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        enrichment = EnrichmentResult(
            lemma="Test",
            gender="das",
            translations=["test"],
        )

        result = await apply_enrichment_to_word(word, enrichment, async_session)
        assert result == "modified"
        assert word.gender == "das"

    @pytest.mark.asyncio
    async def test_apply_enrichment_skipped(self, async_session: AsyncSession):
        """Should return 'skipped' when no changes."""
        from app.routes.vocabulary import apply_enrichment_to_word

        word = Word(lemma="Test", pos="NOUN", gender="das")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        enrichment = EnrichmentResult(
            lemma="Test",
            gender="das",
            translations=None,
        )

        result = await apply_enrichment_to_word(word, enrichment, async_session)
        assert result == "skipped"

    @pytest.mark.asyncio
    async def test_apply_enrichment_flagged_duplicate(self, async_session: AsyncSession):
        """Should return 'flagged' when lemma would be duplicate."""
        from app.routes.vocabulary import apply_enrichment_to_word

        # Create existing word
        existing = Word(lemma="Existing", pos="NOUN")
        async_session.add(existing)
        await async_session.commit()

        # Create word that will be suggested to change to existing lemma
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        enrichment = EnrichmentResult(
            lemma="Existing",  # Same as existing word
            gender="das",
            translations=["test"],
        )

        result = await apply_enrichment_to_word(word, enrichment, async_session)
        assert result == "flagged"
        assert word.needs_review is True
        assert "duplicate" in word.review_reason.lower()

    def test_sse_event_format(self):
        """Should format SSE events correctly."""
        from app.routes.vocabulary import sse_event

        event = sse_event("progress", completed=5, total=10)
        assert 'data: {"type": "progress"' in event
        assert '"completed": 5' in event
        assert '"total": 10' in event
        assert event.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_check_duplicate_lemma(self, async_session: AsyncSession):
        """Should detect duplicate lemmas."""
        from app.routes.vocabulary import check_duplicate_lemma

        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Same lemma and POS but different ID should be duplicate
        is_duplicate = await check_duplicate_lemma(async_session, "Test", "NOUN", exclude_id=999)
        assert is_duplicate is True

        # Same word ID should not be duplicate
        is_duplicate = await check_duplicate_lemma(
            async_session, "Test", "NOUN", exclude_id=word.id
        )
        assert is_duplicate is False

        # Different POS should not be duplicate
        is_duplicate = await check_duplicate_lemma(async_session, "Test", "VERB", exclude_id=999)
        assert is_duplicate is False
