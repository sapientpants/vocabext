"""Document processing command."""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from sqlalchemy import select

from app.cli.utils.async_runner import run_async
from app.cli.utils.console import console, error_console
from app.cli.utils.progress import create_progress, create_simple_progress
from app.config import settings
from app.database import async_session
from app.models import Document, Extraction, Word
from app.services.enricher import Enricher
from app.services.extractor import TextExtractor
from app.services.tokenizer import Tokenizer

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="process",
    help="Document processing commands",
)


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@app.command(name="file")
def process_file(
    file_path: Path = typer.Argument(..., help="Path to document file (PDF, PPTX, TXT, MD, audio)"),
    skip_enrichment: bool = typer.Option(False, "--skip-enrichment", help="Skip LLM enrichment"),
) -> None:
    """Process a document file for vocabulary extraction."""
    # Validate file exists
    if not file_path.exists():
        error_console.print(f"[error]File not found: {file_path}[/]")
        raise typer.Exit(1)

    # Check supported format
    ext = file_path.suffix.lower()
    supported = TextExtractor.supported_extensions()
    if ext not in supported:
        error_console.print(f"[error]Unsupported file format: {ext}[/]")
        error_console.print(f"[dim]Supported formats: {', '.join(sorted(supported))}[/]")
        raise typer.Exit(1)

    run_async(_process_file(file_path, skip_enrichment))


async def _process_file(file_path: Path, skip_enrichment: bool) -> None:
    """Async implementation of process command."""
    console.print(f"\n[bold]Processing:[/] {file_path.name}\n")

    # Compute content hash
    content_hash = _compute_file_hash(file_path)

    async with async_session() as session:
        # Check for duplicate
        existing = await session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        if existing.scalar_one_or_none():
            error_console.print("[warning]This file has already been processed.[/]")
            raise typer.Exit(1)

        # Copy file to upload directory
        dest_path = settings.upload_dir / file_path.name
        if dest_path.exists():
            # Add hash prefix to avoid collision
            dest_path = settings.upload_dir / f"{content_hash[:8]}_{file_path.name}"

        shutil.copy2(file_path, dest_path)

        # Create document record
        document = Document(
            filename=dest_path.name,
            content_hash=content_hash,
            status="processing",
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)

        document_id = document.id
        console.print(f"[dim]Document ID: {document_id}[/]")

        try:
            # Step 1: Extract text
            with create_simple_progress() as progress:
                task = progress.add_task("Extracting text...")
                extractor = TextExtractor(whisper_model=settings.whisper_model)
                raw_text = await extractor.extract(dest_path)
                document.raw_text = raw_text
                await session.commit()
                progress.update(task, description="[green]Text extracted[/]")

            text_preview = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
            console.print(f"[dim]Preview: {text_preview}[/]\n")

            # Step 2: Tokenize
            with create_simple_progress() as progress:
                task = progress.add_task("Tokenizing...")
                tokenizer = Tokenizer(model_name=settings.spacy_model)
                tokens = tokenizer.tokenize(raw_text)
                progress.update(task, description=f"[green]Found {len(tokens)} tokens[/]")

            if not tokens:
                console.print("[warning]No vocabulary tokens found in document.[/]")
                document.status = "pending_review"
                await session.commit()
                return

            # Step 3: Process tokens and create extractions
            console.print(f"\n[info]Processing {len(tokens)} vocabulary items...[/]\n")

            enricher = Enricher() if not skip_enrichment else None
            duplicates = 0
            new_words = 0
            errors = 0

            with create_progress() as progress:
                task = progress.add_task("Processing tokens...", total=len(tokens))

                for token in tokens:
                    # Check if word already exists
                    existing_word = await _find_existing_word(session, token.lemma, token.pos)

                    if existing_word:
                        duplicates += 1
                        extraction = Extraction(
                            document_id=document_id,
                            word_id=existing_word.id,
                            surface_form=token.surface_form,
                            lemma=token.lemma,
                            pos=token.pos,
                            gender=existing_word.gender,
                            plural=existing_word.plural,
                            preterite=existing_word.preterite,
                            past_participle=existing_word.past_participle,
                            auxiliary=existing_word.auxiliary,
                            translations=existing_word.translations,
                            context_sentence=token.context_sentence,
                            status="duplicate",
                        )
                        session.add(extraction)
                        progress.update(
                            task, advance=1, description=f"[dim]Duplicate: {token.lemma}[/]"
                        )
                    else:
                        # Enrich new word via LLM
                        enrichment = None
                        if enricher:
                            try:
                                enrichment = await enricher.enrich(
                                    token.lemma, token.pos, token.context_sentence
                                )
                                if enrichment.error:
                                    errors += 1
                                    logger.warning(
                                        "Enrichment error for '%s': %s",
                                        token.lemma,
                                        enrichment.error,
                                    )
                            except Exception as e:
                                errors += 1
                                logger.error(
                                    "Failed to enrich '%s' (%s): %s",
                                    token.lemma,
                                    token.pos,
                                    e,
                                    exc_info=True,
                                )

                        new_words += 1
                        extraction = Extraction(
                            document_id=document_id,
                            surface_form=token.surface_form,
                            lemma=token.lemma,
                            pos=token.pos,
                            gender=enrichment.gender if enrichment else None,
                            plural=enrichment.plural if enrichment else None,
                            preterite=enrichment.preterite if enrichment else None,
                            past_participle=enrichment.past_participle if enrichment else None,
                            auxiliary=enrichment.auxiliary if enrichment else None,
                            translations=json.dumps(enrichment.translations)
                            if enrichment and enrichment.translations
                            else None,
                            context_sentence=token.context_sentence,
                            status="pending",
                        )
                        session.add(extraction)
                        progress.update(
                            task, advance=1, description=f"[green]New: {token.lemma}[/]"
                        )

                    await session.commit()

            # Update document status
            document.status = "pending_review"
            await session.commit()

            console.print()
            console.print("[success]Processing complete![/]")
            console.print(f"  New words: [green]{new_words}[/]")
            console.print(f"  Duplicates: [dim]{duplicates}[/]")
            if errors:
                console.print(f"  Enrichment errors: [yellow]{errors}[/]")
            console.print()
            console.print(
                f"[info]Run 'vocabext review --document-id {document_id}' to review extractions.[/]"
            )

        except Exception as e:
            error_console.print(f"[error]Processing failed: {e}[/]")
            document.status = "error"
            document.error_message = str(e)
            await session.commit()
            raise typer.Exit(1) from None


async def _find_existing_word(session: "AsyncSession", lemma: str, pos: str) -> Word | None:
    """Find an existing word in vocabulary."""
    stmt = select(Word).where(Word.lemma == lemma, Word.pos == pos)
    result = await session.execute(stmt)
    word: Word | None = result.scalar_one_or_none()
    return word
