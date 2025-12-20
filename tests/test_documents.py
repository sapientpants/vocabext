"""Tests for document routes."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, Extraction


class TestDocumentList:
    """Tests for document list route."""

    @pytest.mark.asyncio
    async def test_list_with_documents(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should list documents."""
        doc1 = Document(filename="test1.pdf", content_hash="abc123")
        doc2 = Document(filename="test2.pdf", content_hash="def456")
        async_session.add_all([doc1, doc2])
        await async_session.commit()

        response = await async_client.get("/documents")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_with_status_filter(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should filter by status."""
        doc1 = Document(filename="test1.pdf", content_hash="abc123", status="processing")
        doc2 = Document(filename="test2.pdf", content_hash="def456", status="pending_review")
        async_session.add_all([doc1, doc2])
        await async_session.commit()

        response = await async_client.get("/documents?status=processing")
        assert response.status_code == 200


class TestDocumentUpload:
    """Tests for document upload route."""

    @pytest.mark.asyncio
    async def test_upload_unsupported_type(self, async_client: AsyncClient):
        """Should reject unsupported file types."""
        content = b"test content"
        files = {"file": ("test.xyz", BytesIO(content), "application/octet-stream")}

        response = await async_client.post("/documents/upload", files=files)
        assert response.status_code == 400
        assert "Unsupported" in response.text


class TestDocumentDetail:
    """Tests for document detail route."""

    @pytest.mark.asyncio
    async def test_get_document_with_extractions(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should show document with extractions."""
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

        response = await async_client.get(f"/documents/{doc.id}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_document_with_status_filter(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Should filter extractions by status."""
        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.flush()

        extraction1 = Extraction(
            document_id=doc.id,
            surface_form="test1",
            lemma="test1",
            pos="NOUN",
            status="pending",
        )
        extraction2 = Extraction(
            document_id=doc.id,
            surface_form="test2",
            lemma="test2",
            pos="NOUN",
            status="accepted",
        )
        async_session.add_all([extraction1, extraction2])
        await async_session.commit()

        response = await async_client.get(f"/documents/{doc.id}?status=pending")
        assert response.status_code == 200


class TestDocumentDelete:
    """Tests for document delete route."""

    @pytest.mark.asyncio
    async def test_delete_document(
        self, async_client: AsyncClient, async_session: AsyncSession, tmp_path: Path
    ):
        """Should delete document and file."""
        # Create a file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"test content")

        doc = Document(filename="test.pdf", content_hash="abc123")
        async_session.add(doc)
        await async_session.commit()

        with patch("app.routes.documents.settings") as mock_settings:
            mock_settings.upload_dir = tmp_path

            response = await async_client.delete(f"/documents/{doc.id}")
            assert response.status_code == 200

        # Document should be deleted
        deleted = await async_session.get(Document, doc.id)
        assert deleted is None
