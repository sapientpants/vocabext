"""Vocabulary browsing and management routes."""

import json
import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Word

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vocabulary", tags=["vocabulary"])


@router.get("", response_class=HTMLResponse)
async def list_vocabulary(
    request: Request,
    search: str = Query("", description="Search by lemma"),
    pos: str = Query("", description="Filter by POS"),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """List all vocabulary words with search and filter."""
    stmt = select(Word).order_by(Word.lemma)

    # Apply search filter
    if search:
        stmt = stmt.where(Word.lemma.ilike(f"%{search}%"))

    # Apply POS filter
    if pos:
        stmt = stmt.where(Word.pos == pos)

    result = await session.execute(stmt)
    words = result.scalars().all()

    # Get available POS values for filter dropdown
    pos_stmt = select(Word.pos).distinct()
    pos_result = await session.execute(pos_stmt)
    pos_values = sorted([p for (p,) in pos_result.all()])

    # Get counts
    count_stmt = select(func.count(Word.id))
    total_count = (await session.execute(count_stmt)).scalar()

    synced_stmt = select(func.count(Word.id)).where(Word.anki_note_id.isnot(None))
    synced_count = (await session.execute(synced_stmt)).scalar()

    return request.app.state.templates.TemplateResponse(
        "vocabulary/list.html",
        {
            "request": request,
            "words": words,
            "search": search,
            "pos": pos,
            "pos_values": pos_values,
            "total_count": total_count,
            "synced_count": synced_count,
        },
    )


@router.get("/{word_id}/edit", response_class=HTMLResponse)
async def edit_word_form(
    request: Request,
    word_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Get the edit form for a word."""
    word = await session.get(Word, word_id)
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    return request.app.state.templates.TemplateResponse(
        "partials/word_edit.html",
        {"request": request, "word": word},
    )


@router.put("/{word_id}", response_class=HTMLResponse)
async def update_word(
    request: Request,
    word_id: int,
    gender: str | None = Form(None),
    plural: str | None = Form(None),
    preterite: str | None = Form(None),
    past_participle: str | None = Form(None),
    auxiliary: str | None = Form(None),
    translations: str = Form(""),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Update a word's metadata."""
    word = await session.get(Word, word_id)
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    # Update fields
    word.gender = gender or None
    word.plural = plural or None
    word.preterite = preterite or None
    word.past_participle = past_participle or None
    word.auxiliary = auxiliary or None

    # Parse translations (comma-separated)
    trans_list = [t.strip() for t in translations.split(",") if t.strip()]
    word.translations = json.dumps(trans_list) if trans_list else None

    # Clear Anki sync status if word was updated
    # (will need to be re-synced)
    if word.anki_note_id:
        word.anki_synced_at = None

    await session.commit()

    return request.app.state.templates.TemplateResponse(
        "partials/word_row.html",
        {"request": request, "word": word},
    )
