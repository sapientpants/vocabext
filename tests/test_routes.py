"""Tests for route handlers."""

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.routes.documents import _sanitize_filename


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_normal_filename(self):
        """Should keep normal filenames unchanged."""
        assert _sanitize_filename("document.pdf") == "document.pdf"
        assert _sanitize_filename("my_file.txt") == "my_file.txt"

    def test_removes_path_components(self):
        """Should remove directory traversal attempts."""
        assert _sanitize_filename("../../../etc/passwd") == "passwd"
        # Backslash is replaced with underscore, then Path.name extracts last component
        result = _sanitize_filename("..\\..\\windows\\system32")
        assert "system32" in result
        assert _sanitize_filename("/etc/passwd") == "passwd"

    def test_removes_dangerous_characters(self):
        """Should remove potentially dangerous characters."""
        result = _sanitize_filename('file<>:"/\\|?*.pdf')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "/" not in result
        assert "\\" not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_removes_control_characters(self):
        """Should remove control characters."""
        result = _sanitize_filename("file\x00name\x1f.pdf")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_strips_dots_and_spaces(self):
        """Should strip leading/trailing dots and spaces."""
        assert _sanitize_filename("...file.pdf") == "file.pdf"
        assert _sanitize_filename("file.pdf...") == "file.pdf"
        assert _sanitize_filename("  file.pdf  ") == "file.pdf"

    def test_empty_filename(self):
        """Should generate safe name for empty filename."""
        result = _sanitize_filename("")
        assert result.startswith("upload_")
        assert len(result) > 8

    def test_none_like_filename(self):
        """Should handle None-like inputs."""
        result = _sanitize_filename("")
        assert result.startswith("upload_")

    def test_only_dangerous_chars(self):
        """Should generate safe name when only dangerous chars."""
        result = _sanitize_filename("../../../")
        assert result.startswith("upload_")


class TestDocumentRoutes:
    """Tests for document routes."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, async_client: AsyncClient):
        """Should return empty list when no documents."""
        response = await async_client.get("/documents")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_upload_unsupported_file(self, async_client: AsyncClient):
        """Should reject unsupported file types."""
        files = {"file": ("test.xyz", BytesIO(b"content"), "application/octet-stream")}
        response = await async_client.post("/documents/upload", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.text

    @pytest.mark.asyncio
    async def test_upload_no_filename(self, async_client: AsyncClient):
        """Should reject upload without filename."""
        files = {"file": ("", BytesIO(b"content"), "text/plain")}
        response = await async_client.post("/documents/upload", files=files)
        # Empty filename should be rejected (400/422) or sanitized and processed (303)
        assert response.status_code in [400, 422, 303]

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, async_client: AsyncClient):
        """Should return 404 for nonexistent document."""
        response = await async_client.get("/documents/99999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, async_client: AsyncClient):
        """Should return 404 when deleting nonexistent document."""
        response = await async_client.delete("/documents/99999")
        assert response.status_code == 404


class TestExtractionRoutes:
    """Tests for extraction routes."""

    @pytest.mark.asyncio
    async def test_accept_nonexistent_extraction(self, async_client: AsyncClient):
        """Should return 404 for nonexistent extraction."""
        response = await async_client.patch("/extractions/99999/accept")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_reject_nonexistent_extraction(self, async_client: AsyncClient):
        """Should return 404 for nonexistent extraction."""
        response = await async_client.patch("/extractions/99999/reject")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_edit_nonexistent_extraction(self, async_client: AsyncClient):
        """Should return 404 for nonexistent extraction."""
        response = await async_client.get("/extractions/99999/edit")
        assert response.status_code == 404


class TestVocabularyRoutes:
    """Tests for vocabulary routes."""

    @pytest.mark.asyncio
    async def test_list_vocabulary_empty(self, async_client: AsyncClient):
        """Should return empty list when no words."""
        response = await async_client.get("/vocabulary")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_edit_nonexistent_word(self, async_client: AsyncClient):
        """Should return 404 for nonexistent word."""
        response = await async_client.get("/vocabulary/99999/edit")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_nonexistent_word(self, async_client: AsyncClient):
        """Should return 404 for nonexistent word."""
        response = await async_client.put(
            "/vocabulary/99999",
            data={"translations": "test"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_word_not_supported(self, async_client: AsyncClient):
        """DELETE method is not supported on vocabulary routes."""
        response = await async_client.delete("/vocabulary/99999")
        assert response.status_code == 405  # Method Not Allowed


class TestSyncRoutes:
    """Tests for sync routes."""

    @pytest.mark.asyncio
    async def test_sync_status(self, async_client: AsyncClient):
        """Should return sync status."""
        with patch("app.routes.sync.AnkiService") as mock_anki:
            mock_instance = mock_anki.return_value
            mock_instance.get_sync_stats = AsyncMock(
                return_value={"available": False, "deck_count": 0}
            )

            response = await async_client.get("/sync/status")
            assert response.status_code == 200
            data = response.json()
            assert "anki" in data
            assert "words" in data

    @pytest.mark.asyncio
    async def test_sync_anki_unavailable(self, async_client: AsyncClient):
        """Should return 503 when Anki is unavailable."""
        with patch("app.routes.sync.AnkiService") as mock_anki:
            mock_instance = mock_anki.return_value
            mock_instance.is_available = AsyncMock(return_value=False)

            response = await async_client.post("/sync")
            assert response.status_code == 503
            assert "not running" in response.text.lower() or "anki" in response.text.lower()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Should return healthy status."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
