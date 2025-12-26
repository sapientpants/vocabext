"""Vocabulary browsing and management routes."""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Word
from app.services.enricher import Enricher, EnrichmentResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vocabulary", tags=["vocabulary"])


def build_diff(word: Word, suggestions: EnrichmentResult) -> dict[str, Any]:
    """Build diff between current word data and LLM suggestions."""
    diff: dict[str, Any] = {}

    # Always compare lemma (include even if LLM returned None, for visibility)
    if suggestions.lemma != word.lemma:
        diff["lemma"] = {"current": word.lemma, "suggested": suggestions.lemma}

    # Determine fields to compare based on POS
    if word.pos == "NOUN":
        fields = ["gender", "plural", "translations"]
    elif word.pos == "VERB":
        fields = ["preterite", "past_participle", "auxiliary", "translations"]
    else:
        fields = ["translations"]

    for field in fields:
        if field == "translations":
            current_val = word.translations_list or []
            suggested_val = suggestions.translations or []
            if current_val != suggested_val:
                diff[field] = {"current": current_val, "suggested": suggested_val}
        else:
            current_val = getattr(word, field)
            suggested_val = getattr(suggestions, field)
            if current_val != suggested_val:
                diff[field] = {"current": current_val, "suggested": suggested_val}

    return diff


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
    lemma: str = Form(...),
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

    # Normalize empty strings to None
    gender = gender or None
    lemma = lemma.strip()

    # Check for unique constraint violation if lemma or gender changed
    if lemma != word.lemma or gender != word.gender:
        stmt = select(Word).where(
            Word.lemma == lemma,
            Word.pos == word.pos,
            Word.id != word_id,
        )
        if gender:
            stmt = stmt.where(Word.gender == gender)
        else:
            stmt = stmt.where(Word.gender.is_(None))

        existing = (await session.execute(stmt)).scalar_one_or_none()
        if existing:
            return request.app.state.templates.TemplateResponse(
                "partials/word_edit.html",
                {
                    "request": request,
                    "word": word,
                    "error": f"A word '{lemma}' with the same POS and gender already exists.",
                },
            )

    # Update fields
    word.lemma = lemma
    word.gender = gender
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


@router.post("/{word_id}/validate", response_class=HTMLResponse)
async def validate_with_llm(
    request: Request,
    word_id: int,
    context_sentence: str = Form(""),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Validate word data with LLM using two-step flow: validate lemma, then enrich."""
    word = await session.get(Word, word_id)
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    # Use the word itself as context if none provided
    context = context_sentence.strip() or word.lemma

    enricher = Enricher()

    # Step 1: Validate the lemma and get corrected form
    lemma_result = await enricher.validate_lemma(word.lemma, word.pos, context)

    if lemma_result.get("error"):
        logger.error(f"Lemma validation failed for '{word.lemma}': {lemma_result['error']}")
        return request.app.state.templates.TemplateResponse(
            "partials/word_diff.html",
            {
                "request": request,
                "word": word,
                "error": lemma_result["error"],
            },
        )

    corrected_lemma = lemma_result.get("corrected_lemma", word.lemma)
    lemma_valid = lemma_result.get("valid", True)
    lemma_reason = lemma_result.get("reason")

    # Step 2: Enrich with the corrected lemma to get grammar details
    result = await enricher.enrich(corrected_lemma, word.pos, context)

    if result.error:
        logger.error(f"LLM enrichment failed for '{corrected_lemma}': {result.error}")
        return request.app.state.templates.TemplateResponse(
            "partials/word_diff.html",
            {
                "request": request,
                "word": word,
                "error": result.error,
            },
        )

    # Override the lemma in result with the validated/corrected one
    result.lemma = corrected_lemma

    # Build diff data
    diff = build_diff(word, result)

    # Add lemma validation info to diff if lemma was corrected
    if not lemma_valid and lemma_reason and "lemma" in diff:
        diff["lemma"]["reason"] = lemma_reason

    # Check if suggested lemma already exists in vocabulary (would be a duplicate)
    duplicate_word = None
    if "lemma" in diff and corrected_lemma and corrected_lemma != word.lemma:
        stmt = select(Word).where(
            Word.lemma == corrected_lemma,
            Word.pos == word.pos,
            Word.id != word.id,
        )
        # For nouns, also match gender
        if word.pos == "NOUN" and result.gender:
            stmt = stmt.where(Word.gender == result.gender)
        duplicate_word = (await session.execute(stmt)).scalar_one_or_none()

    return request.app.state.templates.TemplateResponse(
        "partials/word_diff.html",
        {
            "request": request,
            "word": word,
            "suggestions": result,
            "diff": diff,
            "duplicate_word": duplicate_word,
            "lemma_reason": lemma_reason,
        },
    )


@router.post("/{word_id}/apply-suggestions", response_class=HTMLResponse)
async def apply_suggestions(
    request: Request,
    word_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Apply selected LLM suggestions to word."""
    word = await session.get(Word, word_id)
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    form = await request.form()

    # Check which fields were selected and apply
    field_mapping = {
        "lemma": "lemma",
        "gender": "gender",
        "plural": "plural",
        "preterite": "preterite",
        "past_participle": "past_participle",
        "auxiliary": "auxiliary",
        "translations": "translations",
    }

    updated = False
    for field, attr in field_mapping.items():
        if form.get(f"apply_{field}"):
            suggested_value = str(form.get(f"suggested_{field}", ""))
            if field == "translations":
                trans_list = [t.strip() for t in suggested_value.split(",") if t.strip()]
                word.translations = json.dumps(trans_list) if trans_list else None
            else:
                setattr(word, attr, suggested_value or None)
            updated = True

    # Clear Anki sync status if word was updated
    if updated and word.anki_note_id:
        word.anki_synced_at = None

    await session.commit()

    return request.app.state.templates.TemplateResponse(
        "partials/word_row.html",
        {"request": request, "word": word},
    )


@router.delete("/{word_id}", response_class=HTMLResponse)
async def delete_word(
    word_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Delete a word from vocabulary."""
    word = await session.get(Word, word_id)
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    await session.delete(word)
    await session.commit()

    # Return empty response - HTMX will remove the row
    return Response(status_code=200, content="")
