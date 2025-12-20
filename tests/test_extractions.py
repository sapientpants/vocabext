"""Tests for extraction routes."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, Extraction, Word


class TestExtractionHelpers:
    """Tests for extraction helper functions."""

    @pytest.mark.asyncio
    async def test_update_document_status_marks_reviewed(self, async_session: AsyncSession):
        """Should mark document as reviewed when no pending extractions."""
        from app.routes.extractions import update_document_status

        doc = Document(
            filename="test.pdf",
            content_hash="abc123",
            status="pending_review",
        )
        async_session.add(doc)
        await async_session.flush()

        # Add an accepted extraction (not pending)
        extraction = Extraction(
            document_id=doc.id,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            status="accepted",
        )
        async_session.add(extraction)
        await async_session.commit()

        await update_document_status(doc.id, async_session)

        await async_session.refresh(doc)
        assert doc.status == "reviewed"

    @pytest.mark.asyncio
    async def test_update_document_status_keeps_pending(self, async_session: AsyncSession):
        """Should keep pending_review when pending extractions exist."""
        from app.routes.extractions import update_document_status

        doc = Document(
            filename="test.pdf",
            content_hash="abc123",
            status="pending_review",
        )
        async_session.add(doc)
        await async_session.flush()

        # Add a pending extraction
        extraction = Extraction(
            document_id=doc.id,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            status="pending",
        )
        async_session.add(extraction)
        await async_session.commit()

        await async_session.refresh(doc, ["extractions"])
        await update_document_status(doc.id, async_session)

        await async_session.refresh(doc)
        assert doc.status == "pending_review"

    @pytest.mark.asyncio
    async def test_update_document_status_nonexistent_doc(self, async_session: AsyncSession):
        """Should handle nonexistent document gracefully."""
        from app.routes.extractions import update_document_status

        # Should not raise
        await update_document_status(99999, async_session)


class TestFindOrCreateWord:
    """Tests for _find_or_create_word helper function."""

    @pytest.mark.asyncio
    async def test_finds_existing_word(self, async_session: AsyncSession):
        """Should find existing word with same lemma+pos+gender."""
        from app.routes.extractions import _find_or_create_word

        # Create existing word
        word = Word(lemma="Arbeit", pos="NOUN", gender="die")
        async_session.add(word)
        await async_session.commit()

        # Create extraction matching the word
        extraction = Extraction(
            document_id=1,
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
        )

        result = await _find_or_create_word(extraction, async_session)
        assert result.id == word.id

    @pytest.mark.asyncio
    async def test_creates_new_word(self, async_session: AsyncSession):
        """Should create new word when none exists."""
        from app.routes.extractions import _find_or_create_word

        extraction = Extraction(
            document_id=1,
            surface_form="Test",
            lemma="Test",
            pos="NOUN",
            gender="der",
            plural="Tests",
            translations='["test"]',
        )

        result = await _find_or_create_word(extraction, async_session)
        assert result.id is not None
        assert result.lemma == "Test"
        assert result.pos == "NOUN"
        assert result.gender == "der"

    @pytest.mark.asyncio
    async def test_handles_null_gender(self, async_session: AsyncSession):
        """Should properly handle NULL gender comparison."""
        from app.routes.extractions import _find_or_create_word

        # Create word without gender
        word = Word(lemma="schnell", pos="ADJ", gender=None)
        async_session.add(word)
        await async_session.commit()

        # Create extraction matching (also no gender)
        extraction = Extraction(
            document_id=1,
            surface_form="schnell",
            lemma="schnell",
            pos="ADJ",
            gender=None,
        )

        result = await _find_or_create_word(extraction, async_session)
        assert result.id == word.id


class TestAcceptExtraction:
    """Tests for accept_extraction route."""

    @pytest.mark.asyncio
    async def test_accept_pending_extraction(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should accept a pending extraction."""
        doc = Document(
            filename="test.pdf",
            content_hash="abc123",
            status="pending_review",
        )
        async_session.add(doc)
        await async_session.flush()

        extraction = Extraction(
            document_id=doc.id,
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
            status="pending",
        )
        async_session.add(extraction)
        await async_session.commit()

        response = await async_client.patch(f"/extractions/{extraction.id}/accept")
        assert response.status_code == 200

        await async_session.refresh(extraction)
        assert extraction.status == "accepted"
        assert extraction.word_id is not None

    @pytest.mark.asyncio
    async def test_accept_already_processed(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should reject accepting already processed extraction."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        extraction = Extraction(
            document_id=doc.id,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            status="accepted",  # Already processed
        )
        async_session.add(extraction)
        await async_session.commit()

        response = await async_client.patch(f"/extractions/{extraction.id}/accept")
        assert response.status_code == 400


class TestRejectExtraction:
    """Tests for reject_extraction route."""

    @pytest.mark.asyncio
    async def test_reject_pending_extraction(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should reject a pending extraction."""
        doc = Document(
            filename="test.pdf",
            content_hash="abc123",
            status="pending_review",
        )
        async_session.add(doc)
        await async_session.flush()

        extraction = Extraction(
            document_id=doc.id,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            status="pending",
        )
        async_session.add(extraction)
        await async_session.commit()

        response = await async_client.patch(f"/extractions/{extraction.id}/reject")
        assert response.status_code == 200

        await async_session.refresh(extraction)
        assert extraction.status == "rejected"

    @pytest.mark.asyncio
    async def test_reject_already_processed(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should reject rejecting already processed extraction."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        extraction = Extraction(
            document_id=doc.id,
            surface_form="test",
            lemma="test",
            pos="NOUN",
            status="rejected",  # Already processed
        )
        async_session.add(extraction)
        await async_session.commit()

        response = await async_client.patch(f"/extractions/{extraction.id}/reject")
        assert response.status_code == 400


class TestEditExtractionForm:
    """Tests for edit_extraction_form route."""

    @pytest.mark.asyncio
    async def test_edit_form_nonexistent(self, async_client: AsyncClient):
        """Should return 404 for nonexistent extraction."""
        response = await async_client.get("/extractions/99999/edit")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_edit_form_success(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should return edit form for existing extraction."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        extraction = Extraction(
            document_id=doc.id,
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
            status="pending",
            context_sentence="Die Arbeit ist wichtig.",
        )
        async_session.add(extraction)
        await async_session.commit()

        response = await async_client.get(f"/extractions/{extraction.id}/edit")
        assert response.status_code == 200


class TestUpdateExtraction:
    """Tests for update_extraction route."""

    @pytest.mark.asyncio
    async def test_update_extraction(self, async_client: AsyncClient, async_session: AsyncSession):
        """Should update extraction metadata."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        extraction = Extraction(
            document_id=doc.id,
            surface_form="Arbeit",
            lemma="Arbeit",
            pos="NOUN",
            status="pending",
        )
        async_session.add(extraction)
        await async_session.commit()

        response = await async_client.put(
            f"/extractions/{extraction.id}",
            data={
                "gender": "die",
                "plural": "Arbeiten",
                "translations": "work, job",
            },
        )
        assert response.status_code == 200

        await async_session.refresh(extraction)
        assert extraction.gender == "die"
        assert extraction.plural == "Arbeiten"
        assert "work" in extraction.translations

    @pytest.mark.asyncio
    async def test_update_nonexistent_extraction(self, async_client: AsyncClient):
        """Should return 404 for nonexistent extraction."""
        response = await async_client.put(
            "/extractions/99999",
            data={"translations": "test"},
        )
        assert response.status_code == 404
