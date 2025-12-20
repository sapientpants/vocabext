"""Document upload and management routes."""

import asyncio
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_session
from app.models import Document, Extraction
from app.services.extractor import TextExtractor
from app.tasks.processing import process_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Track running background tasks to prevent orphans on shutdown
_background_tasks: set[asyncio.Task[None]] = set()


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    Removes path components and potentially dangerous characters,
    keeping only the base filename with safe characters.
    """
    if not filename:
        return f"upload_{uuid.uuid4().hex[:8]}"

    # Get only the base filename (removes any directory components)
    name = Path(filename).name

    # Remove any null bytes or control characters
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)

    # Replace potentially dangerous characters
    name = re.sub(r'[<>:"/\\|?*]', "_", name)

    # Remove leading/trailing dots and spaces
    name = name.strip(". ")

    # If nothing left, generate a safe name
    if not name:
        return f"upload_{uuid.uuid4().hex[:8]}"

    return name


@router.get("", response_class=HTMLResponse)
async def list_documents(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """List all documents."""
    stmt = (
        select(Document)
        .options(selectinload(Document.extractions))
        .order_by(Document.created_at.desc())
    )
    result = await session.execute(stmt)
    documents = result.scalars().all()

    return request.app.state.templates.TemplateResponse(
        "documents/list.html",
        {
            "request": request,
            "documents": documents,
            "supported_extensions": TextExtractor.supported_extensions(),
        },
    )


def _create_background_task(coro: Any) -> asyncio.Task[None]:
    """Create a tracked background task that cleans up after itself."""
    task: asyncio.Task[None] = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Upload and process a document."""
    # Validate filename exists
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Sanitize filename to prevent path traversal
    safe_filename = _sanitize_filename(file.filename)
    ext = Path(safe_filename).suffix.lower()

    # Validate file extension
    if ext not in TextExtractor.supported_extensions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    # Read file content with size limit
    max_size = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB",
        )

    # Check for duplicate via content hash
    content_hash = hashlib.sha256(content).hexdigest()

    stmt = select(Document).where(Document.content_hash == content_hash)
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        # Redirect to existing document
        return RedirectResponse(
            url=f"/documents/{existing.id}",
            status_code=303,
        )

    # Save file
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(safe_filename).stem
    file_path = settings.upload_dir / safe_filename

    # Handle filename collisions
    counter = 1
    while file_path.exists():
        file_path = settings.upload_dir / f"{stem}_{counter}{ext}"
        counter += 1

    file_path.write_bytes(content)

    # Create document record
    document = Document(
        filename=file_path.name,
        content_hash=content_hash,
        status="processing",
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    # Start tracked background processing
    _create_background_task(process_document(document.id))

    return RedirectResponse(
        url=f"/documents/{document.id}",
        status_code=303,
    )


@router.get("/{document_id}", response_class=HTMLResponse)
async def review_document(
    request: Request,
    document_id: int,
    show_duplicates: bool = False,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Review extractions for a document."""
    stmt = (
        select(Document)
        .options(selectinload(Document.extractions))
        .where(Document.id == document_id)
    )
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Filter extractions based on show_duplicates flag
    if show_duplicates:
        extractions = document.extractions
    else:
        extractions = [e for e in document.extractions if e.status != "duplicate"]

    # Sort: pending first, then by lemma
    extractions.sort(key=lambda e: (e.status != "pending", e.lemma))

    return request.app.state.templates.TemplateResponse(
        "documents/review.html",
        {
            "request": request,
            "document": document,
            "extractions": extractions,
            "show_duplicates": show_duplicates,
        },
    )


@router.patch("/{document_id}/accept-all", response_class=HTMLResponse)
async def accept_all_pending(
    request: Request,
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Accept all pending extractions for a document."""
    from app.routes.extractions import _accept_extraction, update_document_status

    stmt = select(Extraction).where(
        Extraction.document_id == document_id,
        Extraction.status == "pending",
    )
    result = await session.execute(stmt)
    extractions = result.scalars().all()

    for extraction in extractions:
        await _accept_extraction(extraction, session)

    # Update document status if all reviewed
    await update_document_status(document_id, session)

    # Re-fetch document with updated extractions
    stmt = (
        select(Document)
        .options(selectinload(Document.extractions))
        .where(Document.id == document_id)
    )
    result = await session.execute(stmt)
    document = result.scalar_one()

    extractions = [e for e in document.extractions if e.status != "duplicate"]
    extractions.sort(key=lambda e: (e.status != "pending", e.lemma))

    return request.app.state.templates.TemplateResponse(
        "documents/review.html",
        {
            "request": request,
            "document": document,
            "extractions": extractions,
            "show_duplicates": False,
        },
    )


@router.patch("/{document_id}/reject-all", response_class=HTMLResponse)
async def reject_all_pending(
    request: Request,
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Reject all pending extractions for a document."""
    from app.routes.extractions import update_document_status

    stmt = select(Extraction).where(
        Extraction.document_id == document_id,
        Extraction.status == "pending",
    )
    result = await session.execute(stmt)
    extractions = result.scalars().all()

    for extraction in extractions:
        extraction.status = "rejected"

    await session.commit()

    # Update document status if all reviewed
    await update_document_status(document_id, session)

    # Re-fetch document with updated extractions
    stmt = (
        select(Document)
        .options(selectinload(Document.extractions))
        .where(Document.id == document_id)
    )
    result = await session.execute(stmt)
    document = result.scalar_one()

    extractions = [e for e in document.extractions if e.status != "duplicate"]
    extractions.sort(key=lambda e: (e.status != "pending", e.lemma))

    return request.app.state.templates.TemplateResponse(
        "documents/review.html",
        {
            "request": request,
            "document": document,
            "extractions": extractions,
            "show_duplicates": False,
        },
    )


@router.get("/{document_id}/status", response_class=HTMLResponse)
async def document_status(
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> str:
    """Return just the status badge for htmx polling."""
    stmt = (
        select(Document)
        .options(selectinload(Document.extractions))
        .where(Document.id == document_id)
    )
    result = await session.execute(stmt)
    document = result.scalar_one_or_none()

    if not document:
        return ""

    if document.status == "processing":
        return f"""<span class="status-badge processing"
              hx-get="/documents/{document_id}/status"
              hx-trigger="every 3s"
              hx-swap="outerHTML">
            Processing...
        </span>"""
    elif document.status == "pending_review":
        return '<span class="status-badge pending">Pending Review</span>'
    elif document.status == "reviewed":
        return '<span class="status-badge reviewed">Reviewed</span>'
    elif document.status == "error":
        return f'<span class="status-badge error" title="{document.error_message}">Error</span>'
    return ""


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Reprocess a failed document."""
    document = await session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Reset status and clear error
    document.status = "processing"
    document.error_message = None
    await session.commit()

    # Start tracked background processing
    _create_background_task(process_document(document_id))

    return RedirectResponse(
        url=f"/documents/{document_id}",
        status_code=303,
    )


@router.delete("/{document_id}", response_class=HTMLResponse)
async def delete_document(
    document_id: int,
    session: AsyncSession = Depends(get_session),
) -> str:
    """Delete a document and its uploaded file."""
    document = await session.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete uploaded file
    file_path = settings.upload_dir / document.filename
    if file_path.exists():
        file_path.unlink()

    # Delete document (cascades to extractions)
    await session.delete(document)
    await session.commit()

    # Return empty response for HTMX to swap out the row
    return ""
