"""Vocabulary browsing and management routes."""

import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.responses import StreamingResponse

from app.database import async_session, get_session
from app.models import Word, WordVersion
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


def build_version_diff(word: Word, version: WordVersion) -> dict[str, Any]:
    """Build diff between current word state and a historical version."""
    diff: dict[str, Any] = {}

    # Compare lemma
    if version.lemma != word.lemma:
        diff["lemma"] = {"old": version.lemma, "current": word.lemma}

    # Compare POS (unlikely to change but include for completeness)
    if version.pos != word.pos:
        diff["pos"] = {"old": version.pos, "current": word.pos}

    # Fields based on POS
    if word.pos == "NOUN":
        fields = ["gender", "plural"]
    elif word.pos == "VERB":
        fields = ["preterite", "past_participle", "auxiliary"]
    else:
        fields = []

    for field in fields:
        old_val = getattr(version, field)
        current_val = getattr(word, field)
        if old_val != current_val:
            diff[field] = {"old": old_val, "current": current_val}

    # Compare translations
    old_trans = version.translations_list or []
    current_trans = word.translations_list or []
    if old_trans != current_trans:
        diff["translations"] = {"old": old_trans, "current": current_trans}

    return diff


async def version_exists(session: AsyncSession, word_id: int, version_number: int) -> bool:
    """Check if a version with the given number already exists for this word."""
    stmt = select(WordVersion).where(
        WordVersion.word_id == word_id,
        WordVersion.version_number == version_number,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


def create_version_from_word(word: Word) -> WordVersion:
    """Create a new WordVersion snapshot from current word state."""
    return WordVersion(
        word_id=word.id,
        version_number=word.current_version,
        lemma=word.lemma,
        pos=word.pos,
        gender=word.gender,
        plural=word.plural,
        preterite=word.preterite,
        past_participle=word.past_participle,
        auxiliary=word.auxiliary,
        translations=word.translations,
    )


def word_fields_changed(
    word: Word,
    lemma: str,
    gender: str | None,
    plural: str | None,
    preterite: str | None,
    past_participle: str | None,
    auxiliary: str | None,
    translations_json: str | None,
) -> bool:
    """Check if any word fields have changed."""
    return bool(
        word.lemma != lemma
        or word.gender != gender
        or word.plural != plural
        or word.preterite != preterite
        or word.past_participle != past_participle
        or word.auxiliary != auxiliary
        or word.translations != translations_json
    )


def build_filtered_query(
    search: str = "",
    pos: str = "",
    sync_status: str = "",
    version: str = "",
    updated_within: str = "",
    review_status: str = "",
    random_order: bool = False,
) -> Select[tuple[Word]]:
    """Build word query with filters (shared by list and batch-validate)."""
    stmt = select(Word)
    if random_order:
        stmt = stmt.order_by(func.random())
    else:
        stmt = stmt.order_by(Word.lemma)

    if search:
        stmt = stmt.where(Word.lemma.ilike(f"%{search}%"))
    if pos:
        stmt = stmt.where(Word.pos == pos)
    if sync_status == "synced":
        stmt = stmt.where(
            Word.anki_note_id.isnot(None),
            Word.anki_synced_at.isnot(None),
        )
    elif sync_status == "unsynced":
        stmt = stmt.where(Word.anki_note_id.is_(None))
    elif sync_status == "needs_resync":
        stmt = stmt.where(
            Word.anki_note_id.isnot(None),
            Word.anki_synced_at.is_(None),
        )
    if version == "v1":
        stmt = stmt.where(Word.current_version == 1)
    elif version == "v2+":
        stmt = stmt.where(Word.current_version > 1)
    if updated_within:
        days = int(updated_within)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        stmt = stmt.where(Word.updated_at >= cutoff)
    if review_status == "needs_review":
        stmt = stmt.where(Word.needs_review == True)  # noqa: E712
    elif review_status == "reviewed":
        stmt = stmt.where(Word.needs_review == False)  # noqa: E712

    return stmt


async def check_duplicate_lemma(
    session: AsyncSession, lemma: str, pos: str, exclude_id: int
) -> bool:
    """Check if a word with this lemma and POS exists (excluding current word)."""
    stmt = select(Word.id).where(
        Word.lemma == lemma,
        Word.pos == pos,
        Word.id != exclude_id,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def apply_enrichment_to_word(
    word: Word, enrichment: EnrichmentResult, session: AsyncSession
) -> str:
    """
    Apply LLM enrichment to word, creating version if changed.

    Returns: 'modified', 'flagged', or 'skipped'
    """
    # Check for duplicate lemma - FLAG instead of skip
    if enrichment.lemma and enrichment.lemma != word.lemma:
        if await check_duplicate_lemma(session, enrichment.lemma, word.pos, word.id):
            word.needs_review = True
            word.review_reason = f"LLM suggests duplicate lemma: {enrichment.lemma}"
            return "flagged"

    # Capture old values
    old_values = {
        "lemma": word.lemma,
        "gender": word.gender,
        "plural": word.plural,
        "preterite": word.preterite,
        "past_participle": word.past_participle,
        "auxiliary": word.auxiliary,
        "translations": word.translations,
    }

    # Apply new values (only if provided)
    if enrichment.lemma:
        word.lemma = enrichment.lemma
    if enrichment.gender:
        word.gender = enrichment.gender
    if enrichment.plural:
        word.plural = enrichment.plural
    if enrichment.preterite:
        word.preterite = enrichment.preterite
    if enrichment.past_participle:
        word.past_participle = enrichment.past_participle
    if enrichment.auxiliary:
        word.auxiliary = enrichment.auxiliary
    if enrichment.translations:
        word.translations = json.dumps(enrichment.translations)

    # Check if anything changed
    new_values = {
        "lemma": word.lemma,
        "gender": word.gender,
        "plural": word.plural,
        "preterite": word.preterite,
        "past_participle": word.past_participle,
        "auxiliary": word.auxiliary,
        "translations": word.translations,
    }
    if old_values == new_values:
        return "skipped"

    # Create version snapshot (if doesn't exist)
    if not await version_exists(session, word.id, word.current_version):
        version = WordVersion(
            word_id=word.id,
            version_number=word.current_version,
            lemma=old_values["lemma"],
            pos=word.pos,
            gender=old_values["gender"],
            plural=old_values["plural"],
            preterite=old_values["preterite"],
            past_participle=old_values["past_participle"],
            auxiliary=old_values["auxiliary"],
            translations=old_values["translations"],
        )
        session.add(version)

    word.current_version += 1
    word.anki_synced_at = None
    word.needs_review = False  # Clear any previous review flag
    word.review_reason = None
    return "modified"


def sse_event(event_type: str, **data: Any) -> str:
    """Format an SSE event."""
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n"


@router.get("", response_class=HTMLResponse)
async def list_vocabulary(
    request: Request,
    search: str = Query("", description="Search by lemma"),
    pos: str = Query("", description="Filter by POS"),
    sync_status: str = Query("", description="Filter: synced/unsynced"),
    version: str = Query("", description="Filter: v1/v2+"),
    updated_within: str = Query("", description="Filter: days since update"),
    review_status: str = Query("", description="Filter: needs_review/reviewed"),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """List all vocabulary words with search and filter."""
    stmt = build_filtered_query(search, pos, sync_status, version, updated_within, review_status)

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
            "sync_status": sync_status,
            "version": version,
            "updated_within": updated_within,
            "review_status": review_status,
            "pos_values": pos_values,
            "total_count": total_count,
            "synced_count": synced_count,
        },
    )


@router.get("/batch-validate/stream")
async def batch_validate_stream(  # pragma: no cover
    request: Request,
    search: str = Query("", description="Search by lemma"),
    pos: str = Query("", description="Filter by POS"),
    sync_status: str = Query("", description="Filter: synced/unsynced"),
    version: str = Query("", description="Filter: v1/v2+"),
    updated_within: str = Query("", description="Filter: days since update"),
    review_status: str = Query("", description="Filter: needs_review/reviewed"),
) -> StreamingResponse:
    """Stream batch validation progress via SSE."""

    async def event_generator() -> AsyncGenerator[str, None]:
        enricher = Enricher()

        async with async_session() as session:
            # Build query with same filters as list_vocabulary
            stmt = build_filtered_query(
                search,
                pos,
                sync_status,
                version,
                updated_within,
                review_status,
                random_order=True,
            )
            result = await session.execute(stmt)
            words = result.scalars().all()

            total = len(words)
            modified, skipped, flagged, errors = 0, 0, 0, 0

            for i, word in enumerate(words):
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Batch validation cancelled by client")
                    break

                yield sse_event("progress", completed=i, total=total, word=word.display_word)

                try:
                    # Validate + enrich
                    enrichment = await enricher.validate_and_enrich(word.lemma, word.pos)

                    if enrichment.error:
                        errors += 1
                        yield sse_event("error", word=word.display_word, error=enrichment.error)
                    else:
                        # Apply if changed
                        result_status = await apply_enrichment_to_word(word, enrichment, session)

                        if result_status == "modified":
                            modified += 1
                        elif result_status == "flagged":
                            flagged += 1
                        else:  # "skipped"
                            skipped += 1

                except Exception as e:
                    errors += 1
                    logger.exception(f"Batch validation error for {word.display_word}")
                    yield sse_event("error", word=word.display_word, error=str(e))

                # Commit after EACH word (allows resume on interrupt)
                await session.commit()

            yield sse_event(
                "complete",
                modified=modified,
                skipped=skipped,
                flagged=flagged,
                errors=errors,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/{word_id}", response_class=HTMLResponse)
async def word_detail(
    request: Request,
    word_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Word detail page with edit form and version history."""
    stmt = select(Word).where(Word.id == word_id).options(selectinload(Word.versions))
    result = await session.execute(stmt)
    word = result.scalar_one_or_none()

    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    return request.app.state.templates.TemplateResponse(
        "vocabulary/detail.html",
        {
            "request": request,
            "word": word,
        },
    )


@router.get("/{word_id}/history", response_class=HTMLResponse)
async def word_history(
    request: Request,
    word_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Get version history partial for a word (HTMX)."""
    stmt = select(Word).where(Word.id == word_id).options(selectinload(Word.versions))
    result = await session.execute(stmt)
    word = result.scalar_one_or_none()

    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    return request.app.state.templates.TemplateResponse(
        "partials/version_history.html",
        {
            "request": request,
            "word": word,
        },
    )


@router.get("/{word_id}/compare", response_class=HTMLResponse)
async def compare_versions(
    request: Request,
    word_id: int,
    version: int = Query(..., description="Version number to compare"),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Compare a historical version with current word state."""
    stmt = select(Word).where(Word.id == word_id).options(selectinload(Word.versions))
    result = await session.execute(stmt)
    word = result.scalar_one_or_none()

    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    # Find the requested version
    target_version = None
    for v in word.versions:
        if v.version_number == version:
            target_version = v
            break

    if not target_version:
        raise HTTPException(status_code=404, detail="Version not found")

    diff = build_version_diff(word, target_version)

    return request.app.state.templates.TemplateResponse(
        "partials/version_comparison.html",
        {
            "request": request,
            "word": word,
            "version": target_version,
            "diff": diff,
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
    """Update a word's metadata, creating a new version if changed."""
    stmt = select(Word).where(Word.id == word_id).options(selectinload(Word.versions))
    result = await session.execute(stmt)
    word = result.scalar_one_or_none()

    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    # Normalize empty strings to None
    gender = gender or None
    lemma = lemma.strip()
    plural_val = plural or None
    preterite_val = preterite or None
    past_participle_val = past_participle or None
    auxiliary_val = auxiliary or None

    # Parse translations (comma-separated)
    trans_list = [t.strip() for t in translations.split(",") if t.strip()]
    translations_json = json.dumps(trans_list) if trans_list else None

    # Check for unique constraint violation if lemma or gender changed
    if lemma != word.lemma or gender != word.gender:
        check_stmt = select(Word).where(
            Word.lemma == lemma,
            Word.pos == word.pos,
            Word.id != word_id,
        )
        if gender:
            check_stmt = check_stmt.where(Word.gender == gender)
        else:
            check_stmt = check_stmt.where(Word.gender.is_(None))

        existing = (await session.execute(check_stmt)).scalar_one_or_none()
        if existing:
            # Return just the error div for HTMX to swap in
            return HTMLResponse(
                content=f"<div class=\"edit-error\">A word '{lemma}' with the same POS and gender already exists.</div>",
                status_code=422,
            )

    # Check if any fields have actually changed
    has_changes = word_fields_changed(
        word,
        lemma,
        gender,
        plural_val,
        preterite_val,
        past_participle_val,
        auxiliary_val,
        translations_json,
    )

    if has_changes:
        # Create version snapshot of current state before updating
        # (only if one doesn't already exist from migration)
        if not await version_exists(session, word.id, word.current_version):
            version = create_version_from_word(word)
            session.add(version)

        # Increment version number
        word.current_version += 1

        # Update fields
        word.lemma = lemma
        word.gender = gender
        word.plural = plural_val
        word.preterite = preterite_val
        word.past_participle = past_participle_val
        word.auxiliary = auxiliary_val
        word.translations = translations_json

        # Clear Anki sync status (will need to be re-synced)
        word.anki_synced_at = None

        await session.commit()

    # Return 200 OK - the frontend JS will reload the page
    return HTMLResponse(content="", status_code=200)


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
    """Apply selected LLM suggestions to word, creating a new version."""
    stmt = select(Word).where(Word.id == word_id).options(selectinload(Word.versions))
    result = await session.execute(stmt)
    word = result.scalar_one_or_none()

    if not word:
        raise HTTPException(status_code=404, detail="Word not found")

    form = await request.form()

    # Track old values for change detection
    old_values = {
        "lemma": word.lemma,
        "gender": word.gender,
        "plural": word.plural,
        "preterite": word.preterite,
        "past_participle": word.past_participle,
        "auxiliary": word.auxiliary,
        "translations": word.translations,
    }

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

    for field, attr in field_mapping.items():
        if form.get(f"apply_{field}"):
            suggested_value = str(form.get(f"suggested_{field}", ""))
            if field == "translations":
                trans_list = [t.strip() for t in suggested_value.split(",") if t.strip()]
                word.translations = json.dumps(trans_list) if trans_list else None
            else:
                setattr(word, attr, suggested_value or None)

    # Check if anything actually changed
    has_changes = (
        word.lemma != old_values["lemma"]
        or word.gender != old_values["gender"]
        or word.plural != old_values["plural"]
        or word.preterite != old_values["preterite"]
        or word.past_participle != old_values["past_participle"]
        or word.auxiliary != old_values["auxiliary"]
        or word.translations != old_values["translations"]
    )

    if has_changes:
        # Create version snapshot with old values
        # (only if one doesn't already exist from migration)
        if not await version_exists(session, word.id, word.current_version):
            version = WordVersion(
                word_id=word.id,
                version_number=word.current_version,
                lemma=old_values["lemma"],
                pos=word.pos,
                gender=old_values["gender"],
                plural=old_values["plural"],
                preterite=old_values["preterite"],
                past_participle=old_values["past_participle"],
                auxiliary=old_values["auxiliary"],
                translations=old_values["translations"],
            )
            session.add(version)

        # Increment version number
        word.current_version += 1

        # Clear Anki sync status
        word.anki_synced_at = None

        await session.commit()

    # Redirect to detail page
    return RedirectResponse(
        url=f"/vocabulary/{word_id}",
        status_code=303,
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
