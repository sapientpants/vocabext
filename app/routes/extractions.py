"""Extraction review and management routes."""

import json
import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Document, Extraction, Word

logger = logging.getLogger(__name__)


async def update_document_status(document_id: int, session: AsyncSession) -> None:
    """Update document status based on extraction states."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    stmt = (
        select(Document)
        .options(selectinload(Document.extractions))
        .where(Document.id == document_id)
    )
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()

    if not document:
        return

    # If no pending extractions remain, mark as reviewed
    if document.pending_count == 0 and document.status == "pending_review":
        document.status = "reviewed"
        await session.commit()


router = APIRouter(prefix="/extractions", tags=["extractions"])


async def _accept_extraction(extraction: Extraction, session: AsyncSession) -> Word:
    """
    Accept an extraction and create/link a Word.

    Internal helper used by accept routes.
    """
    if extraction.status not in ("pending",):
        raise ValueError(f"Cannot accept extraction with status: {extraction.status}")

    # Check if word already exists (same lemma+pos+gender)
    stmt = select(Word).where(
        Word.lemma == extraction.lemma,
        Word.pos == extraction.pos,
    )
    if extraction.gender:
        stmt = stmt.where(Word.gender == extraction.gender)

    result = await session.execute(stmt)
    existing_word: Word | None = result.scalar_one_or_none()

    if existing_word is None:
        # Create new word
        new_word = Word(
            lemma=extraction.lemma,
            pos=extraction.pos,
            gender=extraction.gender,
            plural=extraction.plural,
            preterite=extraction.preterite,
            past_participle=extraction.past_participle,
            auxiliary=extraction.auxiliary,
            translations=extraction.translations,
        )
        session.add(new_word)
        await session.flush()
        word: Word = new_word
    else:
        word = existing_word

    # Link extraction to word
    extraction.word_id = word.id
    extraction.status = "accepted"
    await session.commit()

    return word


@router.patch("/{extraction_id}/accept", response_class=HTMLResponse)
async def accept_extraction(
    request: Request,
    extraction_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Accept an extraction, creating a vocabulary word."""
    extraction = await session.get(Extraction, extraction_id)
    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")

    if extraction.status != "pending":
        raise HTTPException(status_code=400, detail="Extraction already processed")

    document_id = extraction.document_id
    await _accept_extraction(extraction, session)
    await update_document_status(document_id, session)

    return request.app.state.templates.TemplateResponse(
        "partials/extraction_row.html",
        {"request": request, "extraction": extraction},
    )


@router.patch("/{extraction_id}/reject", response_class=HTMLResponse)
async def reject_extraction(
    request: Request,
    extraction_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Reject an extraction."""
    extraction = await session.get(Extraction, extraction_id)
    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")

    if extraction.status != "pending":
        raise HTTPException(status_code=400, detail="Extraction already processed")

    document_id = extraction.document_id
    extraction.status = "rejected"
    await session.commit()
    await update_document_status(document_id, session)

    return request.app.state.templates.TemplateResponse(
        "partials/extraction_row.html",
        {"request": request, "extraction": extraction},
    )


@router.get("/{extraction_id}/edit", response_class=HTMLResponse)
async def edit_extraction_form(
    request: Request,
    extraction_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Get the edit form for an extraction."""
    extraction = await session.get(Extraction, extraction_id)
    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")

    return request.app.state.templates.TemplateResponse(
        "partials/extraction_edit.html",
        {"request": request, "extraction": extraction},
    )


@router.put("/{extraction_id}", response_class=HTMLResponse)
async def update_extraction(
    request: Request,
    extraction_id: int,
    gender: str | None = Form(None),
    plural: str | None = Form(None),
    preterite: str | None = Form(None),
    past_participle: str | None = Form(None),
    auxiliary: str | None = Form(None),
    translations: str = Form(""),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Update an extraction's metadata."""
    extraction = await session.get(Extraction, extraction_id)
    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found")

    # Update fields
    extraction.gender = gender or None
    extraction.plural = plural or None
    extraction.preterite = preterite or None
    extraction.past_participle = past_participle or None
    extraction.auxiliary = auxiliary or None

    # Parse translations (comma-separated)
    trans_list = [t.strip() for t in translations.split(",") if t.strip()]
    extraction.translations = json.dumps(trans_list) if trans_list else None

    await session.commit()

    return request.app.state.templates.TemplateResponse(
        "partials/extraction_row.html",
        {"request": request, "extraction": extraction},
    )
