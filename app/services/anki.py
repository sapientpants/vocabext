"""AnkiConnect service for syncing vocabulary to Anki."""

import logging
from typing import Any

import httpx

from app.config import settings
from app.models import Word

logger = logging.getLogger(__name__)


class AnkiService:
    """Sync vocabulary to Anki via AnkiConnect."""

    def __init__(
        self,
        url: str | None = None,
        deck: str | None = None,
        note_type: str | None = None,
    ) -> None:
        self.url = url or settings.anki_connect_url
        self.deck = deck or settings.anki_deck
        self.note_type = note_type or settings.anki_note_type

    async def _invoke(self, action: str, **params: Any) -> Any:
        """Invoke an AnkiConnect action."""
        payload = {
            "action": action,
            "version": 6,
            "params": params,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()

        if result.get("error"):
            raise Exception(f"AnkiConnect error: {result['error']}")

        return result.get("result")

    async def is_available(self) -> bool:
        """Check if AnkiConnect is available."""
        try:
            await self._invoke("version")
            return True
        except Exception as e:
            logger.warning(f"AnkiConnect not available: {e}")
            return False

    async def ensure_note_type(self) -> None:
        """Ensure the German Vocabulary note type exists with correct fields."""
        model_names = await self._invoke("modelNames")

        if self.note_type in model_names:
            logger.info(f"Note type '{self.note_type}' already exists")
            # Remove Context field if it exists
            try:
                field_names = await self._invoke("modelFieldNames", modelName=self.note_type)
                if "Context" in field_names:
                    logger.info(f"Removing Context field from '{self.note_type}'")
                    await self._invoke(
                        "modelFieldRemove", modelName=self.note_type, fieldName="Context"
                    )
            except Exception as e:
                logger.warning(f"Could not check/remove Context field: {e}")
            return

        logger.info(f"Creating note type '{self.note_type}'")

        await self._invoke(
            "createModel",
            modelName=self.note_type,
            inOrderFields=["Front", "Back", "Grammar"],
            css=""".card {
    font-family: arial;
    font-size: 20px;
    text-align: center;
    color: black;
    background-color: white;
}
.grammar {
    font-size: 14px;
    color: #666;
    margin-top: 10px;
}""",
            cardTemplates=[
                {
                    "Name": "Card 1",
                    "Front": "{{Front}}",
                    "Back": """{{FrontSide}}
<hr id=answer>
{{Back}}
<div class="grammar">{{Grammar}}</div>""",
                }
            ],
        )

    async def ensure_deck(self) -> None:
        """Ensure the target deck exists."""
        await self._invoke("createDeck", deck=self.deck)

    async def add_note(self, word: Word) -> int:
        """
        Add a word to Anki as a new note.

        Returns the note ID.
        """
        translations_text = ", ".join(word.translations_list)

        note = {
            "deckName": self.deck,
            "modelName": self.note_type,
            "fields": {
                "Front": word.display_word,
                "Back": translations_text,
                "Grammar": word.grammar_info,
            },
            "options": {
                "allowDuplicate": False,
            },
            "tags": [f"pos:{word.pos.lower()}", "vocabext"],
        }

        note_id: int = await self._invoke("addNote", note=note)
        logger.info(f"Added note {note_id} for word '{word.lemma}'")
        return note_id

    async def update_note(self, word: Word) -> None:
        """Update an existing Anki note."""
        if not word.anki_note_id:
            raise ValueError("Word has no Anki note ID")

        translations_text = ", ".join(word.translations_list)

        await self._invoke(
            "updateNoteFields",
            note={
                "id": word.anki_note_id,
                "fields": {
                    "Front": word.display_word,
                    "Back": translations_text,
                    "Grammar": word.grammar_info,
                },
            },
        )
        logger.info(f"Updated note {word.anki_note_id} for word '{word.lemma}'")

    async def find_existing_note(self, word: Word) -> int | None:
        """Find an existing note in Anki for this word."""
        # Search by the Front field (display_word) in our note type
        query = f'"note:{self.note_type}" "Front:{word.display_word}"'
        note_ids: list[int] = await self._invoke("findNotes", query=query)

        if note_ids:
            return note_ids[0]
        return None

    async def note_exists(self, note_id: int) -> bool:
        """Check if a note exists in Anki."""
        try:
            notes_info = await self._invoke("notesInfo", notes=[note_id])
            return bool(notes_info and notes_info[0].get("noteId"))
        except Exception:
            return False

    async def sync_word(self, word: Word) -> int | None:
        """
        Sync a word to Anki (create or update).

        Returns the note ID, or None if sync failed.
        """
        try:
            # If we have a note ID, verify it still exists in Anki
            if word.anki_note_id:
                if await self.note_exists(word.anki_note_id):
                    await self.update_note(word)
                    return int(word.anki_note_id)
                # Note was deleted from Anki, clear our reference
                logger.info(f"Note {word.anki_note_id} no longer exists for '{word.lemma}'")

            # Check if note already exists in Anki (by Front field)
            existing_id = await self.find_existing_note(word)
            if existing_id:
                logger.info(f"Found existing note {existing_id} for word '{word.lemma}'")
                # Update the existing note with our data
                word.anki_note_id = existing_id
                await self.update_note(word)
                return existing_id

            # Create new note
            return await self.add_note(word)
        except Exception as e:
            logger.error(f"Failed to sync word '{word.lemma}': {e}")
            return None

    async def get_sync_stats(self) -> dict[str, Any]:
        """Get sync statistics."""
        try:
            if not await self.is_available():
                return {"available": False}

            deck_names = await self._invoke("deckNames")
            has_deck = self.deck in deck_names

            return {
                "available": True,
                "deck_exists": has_deck,
                "deck_name": self.deck,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def get_all_note_ids(self) -> list[int]:
        """Get all note IDs in the vocabulary deck."""
        query = f'"deck:{self.deck}" "note:{self.note_type}"'
        note_ids: list[int] = await self._invoke("findNotes", query=query)
        return note_ids

    async def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes from Anki."""
        if not note_ids:
            return
        await self._invoke("deleteNotes", notes=note_ids)
        logger.info(f"Deleted {len(note_ids)} orphaned notes from Anki")
