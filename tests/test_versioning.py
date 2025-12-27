"""Tests for word versioning functionality."""

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Word, WordVersion


class TestWordVersion:
    """Tests for WordVersion model."""

    @pytest.mark.asyncio
    async def test_create_version(self, async_session):
        """Should create a version for a word."""
        word = Word(
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
            plural="Arbeiten",
            translations=json.dumps(["work"]),
        )
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        version = WordVersion(
            word_id=word.id,
            version_number=1,
            lemma=word.lemma,
            pos=word.pos,
            gender=word.gender,
            plural=word.plural,
            translations=word.translations,
        )
        async_session.add(version)
        await async_session.commit()

        stmt = select(WordVersion).where(WordVersion.word_id == word.id)
        result = await async_session.execute(stmt)
        versions = result.scalars().all()

        assert len(versions) == 1
        assert versions[0].lemma == "Arbeit"
        assert versions[0].version_number == 1

    @pytest.mark.asyncio
    async def test_version_translations_list(self, async_session):
        """Should parse translations JSON correctly."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        version = WordVersion(
            word_id=word.id,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            translations=json.dumps(["test", "exam"]),
        )
        async_session.add(version)
        await async_session.commit()
        await async_session.refresh(version)

        assert version.translations_list == ["test", "exam"]

    @pytest.mark.asyncio
    async def test_version_translations_list_empty(self, async_session):
        """Should return empty list for null translations."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        version = WordVersion(
            word_id=word.id,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            translations=None,
        )
        async_session.add(version)
        await async_session.commit()
        await async_session.refresh(version)

        assert version.translations_list == []

    @pytest.mark.asyncio
    async def test_cascade_delete(self, async_session):
        """Should delete versions when word is deleted."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        for i in range(3):
            version = WordVersion(
                word_id=word.id,
                version_number=i + 1,
                lemma="Test",
                pos="NOUN",
            )
            async_session.add(version)
        await async_session.commit()

        # Verify versions exist
        stmt = select(WordVersion).where(WordVersion.word_id == word.id)
        result = await async_session.execute(stmt)
        assert len(result.scalars().all()) == 3

        # Delete word
        await async_session.delete(word)
        await async_session.commit()

        # Verify versions are deleted
        result = await async_session.execute(stmt)
        assert len(result.scalars().all()) == 0


class TestWordNeedsSync:
    """Tests for Word.needs_sync property."""

    @pytest.mark.asyncio
    async def test_needs_sync_never_synced(self, async_session):
        """Should need sync if never synced to Anki."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        assert word.needs_sync is True

    @pytest.mark.asyncio
    async def test_needs_sync_no_versions(self, async_session):
        """Should need sync if synced but no versions."""
        word = Word(
            lemma="Test",
            pos="NOUN",
            anki_note_id=12345,
            anki_synced_at=datetime.now(timezone.utc),
        )
        async_session.add(word)
        await async_session.commit()

        # Load with versions
        stmt = select(Word).where(Word.id == word.id).options(selectinload(Word.versions))
        result = await async_session.execute(stmt)
        word = result.scalar_one()

        assert word.needs_sync is True

    @pytest.mark.asyncio
    async def test_needs_sync_version_after_sync(self, async_session):
        """Should need sync if version created after last sync."""
        # Use offset-naive datetimes to match SQLite storage
        sync_time = datetime(2024, 1, 1, 12, 0, 0)
        word = Word(
            lemma="Test",
            pos="NOUN",
            anki_note_id=12345,
            anki_synced_at=sync_time,
        )
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Add a version created after sync
        version = WordVersion(
            word_id=word.id,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            created_at=datetime(2024, 1, 2, 12, 0, 0),
        )
        async_session.add(version)
        await async_session.commit()

        # Load with versions
        stmt = select(Word).where(Word.id == word.id).options(selectinload(Word.versions))
        result = await async_session.execute(stmt)
        word = result.scalar_one()

        assert word.needs_sync is True

    @pytest.mark.asyncio
    async def test_needs_sync_version_before_sync(self, async_session):
        """Should not need sync if version created before last sync."""
        # Use offset-naive datetimes to match SQLite storage
        version_time = datetime(2024, 1, 1, 12, 0, 0)
        sync_time = datetime(2024, 1, 2, 12, 0, 0)

        word = Word(
            lemma="Test",
            pos="NOUN",
            anki_note_id=12345,
            anki_synced_at=sync_time,
        )
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Add a version created before sync
        version = WordVersion(
            word_id=word.id,
            version_number=1,
            lemma="Test",
            pos="NOUN",
            created_at=version_time,
        )
        async_session.add(version)
        await async_session.commit()

        # Load with versions
        stmt = select(Word).where(Word.id == word.id).options(selectinload(Word.versions))
        result = await async_session.execute(stmt)
        word = result.scalar_one()

        assert word.needs_sync is False


class TestVersionRoutes:
    """Tests for versioning routes."""

    @pytest.mark.asyncio
    async def test_word_detail_page(self, client, async_session):
        """Should render word detail page."""
        word = Word(lemma="Test", pos="NOUN", gender="das")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        response = client.get(f"/vocabulary/{word.id}")
        assert response.status_code == 200
        assert "Test" in response.text
        assert "Version 1" in response.text

    @pytest.mark.asyncio
    async def test_word_detail_not_found(self, client):
        """Should return 404 for non-existent word."""
        response = client.get("/vocabulary/99999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_version_history_endpoint(self, client, async_session):
        """Should return version history partial."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Add a version
        version = WordVersion(
            word_id=word.id,
            version_number=1,
            lemma="Test",
            pos="NOUN",
        )
        async_session.add(version)
        await async_session.commit()

        response = client.get(f"/vocabulary/{word.id}/history")
        assert response.status_code == 200
        assert "v1" in response.text

    @pytest.mark.asyncio
    async def test_compare_versions_endpoint(self, client, async_session):
        """Should return version comparison partial."""
        word = Word(
            lemma="Updated",
            pos="NOUN",
            gender="der",
            current_version=2,
        )
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        # Add old version with different data
        version = WordVersion(
            word_id=word.id,
            version_number=1,
            lemma="Original",
            pos="NOUN",
            gender="die",
        )
        async_session.add(version)
        await async_session.commit()

        response = client.get(f"/vocabulary/{word.id}/compare?version=1")
        assert response.status_code == 200
        assert "Original" in response.text
        assert "Updated" in response.text

    @pytest.mark.asyncio
    async def test_compare_version_not_found(self, client, async_session):
        """Should return 404 for non-existent version."""
        word = Word(lemma="Test", pos="NOUN")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        response = client.get(f"/vocabulary/{word.id}/compare?version=99")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_creates_version(self, client, async_session):
        """Should create version on update."""
        word = Word(lemma="Test", pos="NOUN", gender="das")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        response = client.put(
            f"/vocabulary/{word.id}",
            data={
                "lemma": "Updated",
                "gender": "der",
                "translations": "test",
            },
            follow_redirects=False,
        )

        # Should redirect to detail page
        assert response.status_code == 303
        assert f"/vocabulary/{word.id}" in response.headers["location"]

        # Verify version was created
        stmt = select(WordVersion).where(WordVersion.word_id == word.id)
        result = await async_session.execute(stmt)
        versions = result.scalars().all()

        assert len(versions) == 1
        assert versions[0].lemma == "Test"  # Old value
        assert versions[0].gender == "das"  # Old value

        # Verify word was updated
        await async_session.refresh(word)
        assert word.lemma == "Updated"
        assert word.gender == "der"
        assert word.current_version == 2

    @pytest.mark.asyncio
    async def test_update_no_changes_no_version(self, client, async_session):
        """Should not create version if no changes."""
        word = Word(lemma="Test", pos="NOUN", gender="das")
        async_session.add(word)
        await async_session.commit()
        await async_session.refresh(word)

        response = client.put(
            f"/vocabulary/{word.id}",
            data={
                "lemma": "Test",
                "gender": "das",
                "translations": "",
            },
            follow_redirects=False,
        )

        assert response.status_code == 303

        # Verify no version was created
        stmt = select(WordVersion).where(WordVersion.word_id == word.id)
        result = await async_session.execute(stmt)
        versions = result.scalars().all()

        assert len(versions) == 0
        assert word.current_version == 1
