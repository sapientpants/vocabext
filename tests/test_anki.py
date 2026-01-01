"""Tests for Anki service."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models import Word
from app.services.anki import AnkiService


class TestAnkiServiceInit:
    """Tests for AnkiService initialization."""

    def test_default_settings(self):
        """Should use default settings."""
        with patch("app.services.anki.settings") as mock_settings:
            mock_settings.anki_connect_url = "http://localhost:8765"
            mock_settings.anki_deck = "German Vocabulary"
            mock_settings.anki_note_type = "German Vocabulary"

            service = AnkiService()
            assert service.url == "http://localhost:8765"
            assert service.deck == "German Vocabulary"
            assert service.note_type == "German Vocabulary"

    def test_custom_settings(self):
        """Should accept custom settings."""
        service = AnkiService(
            url="http://custom:9999",
            deck="Custom Deck",
            note_type="Custom Note",
        )
        assert service.url == "http://custom:9999"
        assert service.deck == "Custom Deck"
        assert service.note_type == "Custom Note"


class TestAnkiServiceInvoke:
    """Tests for AnkiService._invoke method."""

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """Should successfully invoke AnkiConnect action."""
        service = AnkiService(url="http://test:8765")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.post.return_value = AsyncMock(json=lambda: {"result": "6", "error": None})
            mock_instance.post.return_value.raise_for_status = lambda: None
            mock_client.return_value = mock_instance

            result = await service._invoke("version")
            assert result == "6"

    @pytest.mark.asyncio
    async def test_invoke_error(self):
        """Should raise exception on AnkiConnect error."""
        service = AnkiService(url="http://test:8765")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.post.return_value = AsyncMock(
                json=lambda: {"result": None, "error": "Some error"}
            )
            mock_instance.post.return_value.raise_for_status = lambda: None
            mock_client.return_value = mock_instance

            with pytest.raises(Exception, match="AnkiConnect error"):
                await service._invoke("badAction")


class TestAnkiServiceIsAvailable:
    """Tests for AnkiService.is_available method."""

    @pytest.mark.asyncio
    async def test_available_when_responding(self):
        """Should return True when AnkiConnect responds."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock(return_value="6")

        result = await service.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_unavailable_on_error(self):
        """Should return False when AnkiConnect errors."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock(side_effect=Exception("Connection refused"))

        result = await service.is_available()
        assert result is False


class TestAnkiServiceEnsureNoteType:
    """Tests for AnkiService.ensure_note_type method."""

    @pytest.mark.asyncio
    async def test_creates_note_type_when_missing(self):
        """Should create note type if it doesn't exist."""
        service = AnkiService(url="http://test:8765", note_type="Test Note")
        service._invoke = AsyncMock(return_value=[])

        await service.ensure_note_type()

        # Should have called createModel
        calls = service._invoke.call_args_list
        assert any(call[0][0] == "createModel" for call in calls)

    @pytest.mark.asyncio
    async def test_skips_creation_when_exists(self):
        """Should not create note type if it already exists."""
        service = AnkiService(url="http://test:8765", note_type="Test Note")
        service._invoke = AsyncMock(side_effect=[["Test Note"], ["Front", "Back", "Grammar"]])

        await service.ensure_note_type()

        # Should not have called createModel
        calls = service._invoke.call_args_list
        assert not any(call[0][0] == "createModel" for call in calls)

    @pytest.mark.asyncio
    async def test_removes_context_field_when_exists(self):
        """Should remove Context field if it exists in the model."""
        service = AnkiService(url="http://test:8765", note_type="Test Note")
        # First call: modelNames returns our model
        # Second call: modelFieldNames returns fields including Context
        # Third call: modelFieldRemove
        service._invoke = AsyncMock(
            side_effect=[["Test Note"], ["Front", "Back", "Grammar", "Context"], None]
        )

        await service.ensure_note_type()

        # Should have called modelFieldRemove
        calls = service._invoke.call_args_list
        assert any(call[0][0] == "modelFieldRemove" for call in calls)

    @pytest.mark.asyncio
    async def test_handles_context_field_check_error(self):
        """Should handle error when checking Context field."""
        service = AnkiService(url="http://test:8765", note_type="Test Note")
        # First call: modelNames returns our model
        # Second call: modelFieldNames raises
        service._invoke = AsyncMock(side_effect=[["Test Note"], Exception("API error")])

        # Should not raise
        await service.ensure_note_type()


class TestAnkiServiceEnsureDeck:
    """Tests for AnkiService.ensure_deck method."""

    @pytest.mark.asyncio
    async def test_creates_deck(self):
        """Should create deck."""
        service = AnkiService(url="http://test:8765", deck="Test Deck")
        service._invoke = AsyncMock()

        await service.ensure_deck()

        service._invoke.assert_called_once_with("createDeck", deck="Test Deck")


class TestAnkiServiceAddNote:
    """Tests for AnkiService.add_note method."""

    @pytest.mark.asyncio
    async def test_add_note_success(self):
        """Should add note and return note ID."""
        service = AnkiService(
            url="http://test:8765",
            deck="Test Deck",
            note_type="Test Note",
        )
        service._invoke = AsyncMock(return_value=12345)

        word = Word(
            lemma="Arbeit",
            pos="NOUN",
            gender="die",
            plural="Arbeiten",
            translations='["work", "job"]',
        )

        note_id = await service.add_note(word)

        assert note_id == 12345
        service._invoke.assert_called_once()
        call_args = service._invoke.call_args
        assert call_args[0][0] == "addNote"


class TestAnkiServiceUpdateNote:
    """Tests for AnkiService.update_note method."""

    @pytest.mark.asyncio
    async def test_update_note_success(self):
        """Should update existing note."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock()

        word = Word(
            lemma="Arbeit",
            pos="NOUN",
            anki_note_id=12345,
            translations='["work"]',
        )

        await service.update_note(word)

        service._invoke.assert_called_once()
        call_args = service._invoke.call_args
        assert call_args[0][0] == "updateNoteFields"

    @pytest.mark.asyncio
    async def test_update_note_no_id(self):
        """Should raise error when word has no note ID."""
        service = AnkiService(url="http://test:8765")

        word = Word(
            lemma="Arbeit",
            pos="NOUN",
            anki_note_id=None,
        )

        with pytest.raises(ValueError, match="no Anki note ID"):
            await service.update_note(word)


class TestAnkiServiceFindExistingNote:
    """Tests for AnkiService.find_existing_note method."""

    @pytest.mark.asyncio
    async def test_finds_existing_note(self):
        """Should return note ID when found."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock(return_value=[12345])

        word = Word(lemma="Arbeit", pos="NOUN", gender="die")
        result = await service.find_existing_note(word)

        assert result == 12345

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        """Should return None when note not found."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock(return_value=[])

        word = Word(lemma="Arbeit", pos="NOUN")
        result = await service.find_existing_note(word)

        assert result is None


class TestAnkiServiceNoteExists:
    """Tests for AnkiService.note_exists method."""

    @pytest.mark.asyncio
    async def test_exists_returns_true(self):
        """Should return True when note exists."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock(return_value=[{"noteId": 12345}])

        result = await service.note_exists(12345)
        assert result is True

    @pytest.mark.asyncio
    async def test_not_exists_returns_false(self):
        """Should return False when note doesn't exist."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock(return_value=[{}])

        result = await service.note_exists(12345)
        assert result is False

    @pytest.mark.asyncio
    async def test_error_returns_false(self):
        """Should return False on error."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock(side_effect=Exception("Error"))

        result = await service.note_exists(12345)
        assert result is False


class TestAnkiServiceSyncWord:
    """Tests for AnkiService.sync_word method."""

    @pytest.mark.asyncio
    async def test_sync_creates_new_note(self):
        """Should create new note when word has no note ID."""
        service = AnkiService(url="http://test:8765")
        service.find_existing_note = AsyncMock(return_value=None)
        service.add_note = AsyncMock(return_value=12345)

        word = Word(lemma="Arbeit", pos="NOUN")
        result = await service.sync_word(word)

        assert result == 12345
        service.add_note.assert_called_once_with(word)

    @pytest.mark.asyncio
    async def test_sync_updates_existing_note(self):
        """Should update note when word has valid note ID."""
        service = AnkiService(url="http://test:8765")
        service.note_exists = AsyncMock(return_value=True)
        service.update_note = AsyncMock()

        word = Word(lemma="Arbeit", pos="NOUN", anki_note_id=12345)
        result = await service.sync_word(word)

        assert result == 12345
        service.update_note.assert_called_once_with(word)

    @pytest.mark.asyncio
    async def test_sync_recreates_deleted_note(self):
        """Should create new note when existing note was deleted."""
        service = AnkiService(url="http://test:8765")
        service.note_exists = AsyncMock(return_value=False)
        service.find_existing_note = AsyncMock(return_value=None)
        service.add_note = AsyncMock(return_value=99999)

        word = Word(lemma="Arbeit", pos="NOUN", anki_note_id=12345)
        result = await service.sync_word(word)

        assert result == 99999

    @pytest.mark.asyncio
    async def test_sync_handles_error(self):
        """Should return None on sync error."""
        service = AnkiService(url="http://test:8765")
        service.find_existing_note = AsyncMock(side_effect=Exception("Error"))

        word = Word(lemma="Arbeit", pos="NOUN")
        result = await service.sync_word(word)

        assert result is None


class TestAnkiServiceGetSyncStats:
    """Tests for AnkiService.get_sync_stats method."""

    @pytest.mark.asyncio
    async def test_stats_when_available(self):
        """Should return stats when AnkiConnect is available."""
        service = AnkiService(url="http://test:8765", deck="Test Deck")
        service.is_available = AsyncMock(return_value=True)
        service._invoke = AsyncMock(return_value=["Test Deck", "Other Deck"])

        result = await service.get_sync_stats()

        assert result["available"] is True
        assert result["deck_exists"] is True
        assert result["deck_name"] == "Test Deck"

    @pytest.mark.asyncio
    async def test_stats_when_unavailable(self):
        """Should return unavailable when AnkiConnect not responding."""
        service = AnkiService(url="http://test:8765")
        service.is_available = AsyncMock(return_value=False)

        result = await service.get_sync_stats()

        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_stats_on_error(self):
        """Should return error info on exception."""
        service = AnkiService(url="http://test:8765")
        service.is_available = AsyncMock(side_effect=Exception("Connection error"))

        result = await service.get_sync_stats()

        assert result["available"] is False
        assert "error" in result


class TestAnkiServiceSyncWordExisting:
    """Tests for sync_word finding existing notes."""

    @pytest.mark.asyncio
    async def test_sync_finds_and_updates_existing_by_front(self):
        """Should find existing note by Front field and update it."""
        service = AnkiService(url="http://test:8765")
        service.note_exists = AsyncMock(return_value=False)  # No stored ID
        service.find_existing_note = AsyncMock(return_value=54321)  # Found by Front
        service.update_note = AsyncMock()

        word = Word(lemma="Arbeit", pos="NOUN")  # No anki_note_id
        result = await service.sync_word(word)

        assert result == 54321
        assert word.anki_note_id == 54321
        service.update_note.assert_called_once_with(word)


class TestAnkiServiceGetAllNoteIds:
    """Tests for get_all_note_ids method."""

    @pytest.mark.asyncio
    async def test_get_all_note_ids(self):
        """Should return all note IDs for deck and note type."""
        service = AnkiService(
            url="http://test:8765",
            deck="Test Deck",
            note_type="Test Note",
        )
        service._invoke = AsyncMock(return_value=[123, 456, 789])

        result = await service.get_all_note_ids()

        assert result == [123, 456, 789]
        service._invoke.assert_called_once()
        call_args = service._invoke.call_args
        assert call_args[0][0] == "findNotes"
        assert "deck:Test Deck" in call_args[1]["query"]
        assert "note:Test Note" in call_args[1]["query"]


class TestAnkiServiceDeleteNotes:
    """Tests for delete_notes method."""

    @pytest.mark.asyncio
    async def test_delete_notes(self):
        """Should delete specified notes."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock()

        await service.delete_notes([123, 456])

        service._invoke.assert_called_once_with("deleteNotes", notes=[123, 456])

    @pytest.mark.asyncio
    async def test_delete_notes_empty_list(self):
        """Should not call API with empty list."""
        service = AnkiService(url="http://test:8765")
        service._invoke = AsyncMock()

        await service.delete_notes([])

        service._invoke.assert_not_called()
