"""Anki sync routes."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_session
from app.models import Word
from app.services.anki import AnkiService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])


@router.post("", response_class=HTMLResponse)
async def sync_all(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Sync all unsynced words to Anki."""
    anki = AnkiService()

    # Check if Anki is available
    if not await anki.is_available():
        return HTMLResponse(
            content="""<div class="alert alert-error">
                Anki is not running or AnkiConnect is not installed.
                Please start Anki and try again.
            </div>""",
            status_code=200,
        )

    # Ensure deck and note type exist
    try:
        await anki.ensure_deck()
        await anki.ensure_note_type()
    except Exception as e:
        logger.error(f"Failed to setup Anki: {e}")
        return HTMLResponse(
            content=f'<div class="alert alert-error">Failed to setup Anki: {e}</div>',
            status_code=200,
        )

    # Get all words for syncing
    stmt = select(Word).options(selectinload(Word.extractions))
    result = await session.execute(stmt)
    words = result.scalars().all()

    synced = 0
    failed = 0

    for word in words:
        note_id = await anki.sync_word(word)
        if note_id:
            word.anki_note_id = note_id
            word.anki_synced_at = datetime.utcnow()
            synced += 1
        else:
            failed += 1

    await session.commit()

    if failed:
        return HTMLResponse(
            content=f"""<div class="alert alert-warning">
                Synced {synced} words to Anki. {failed} failed.
            </div>""",
            status_code=200,
        )

    return HTMLResponse(
        content=f"""<div class="alert alert-success">
            Successfully synced {synced} words to Anki.
        </div>""",
        status_code=200,
    )


@router.get("/status", response_class=JSONResponse)
async def sync_status(
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Get sync status and stats."""
    anki = AnkiService()
    anki_stats = await anki.get_sync_stats()

    # Count words
    total_stmt = select(Word)
    result = await session.execute(total_stmt)
    total = len(result.scalars().all())

    synced_stmt = select(Word).where(Word.anki_note_id.isnot(None))
    result = await session.execute(synced_stmt)
    synced = len(result.scalars().all())

    return {
        "anki": anki_stats,
        "words": {
            "total": total,
            "synced": synced,
            "unsynced": total - synced,
        },
    }
