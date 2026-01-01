"""Document processing command."""

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from sqlalchemy import select

from app.cli.utils.async_runner import run_async
from app.cli.utils.console import console, error_console
from app.cli.utils.progress import create_progress, create_simple_progress
from app.config import settings
from app.database import async_session
from app.models import Word
from app.services.enricher import Enricher, EnrichmentResult
from app.services.extractor import TextExtractor
from app.services.tokenizer import TokenInfo, Tokenizer

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="process",
    help="Document processing commands",
)


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


async def _enrich_token(
    enricher: Enricher, token: TokenInfo
) -> tuple[TokenInfo, EnrichmentResult | None]:
    """Enrich a single token with error handling."""
    try:
        enrichment = await enricher.enrich(token.lemma, token.pos, token.context_sentence)
        return token, enrichment
    except Exception as e:
        logger.error("Failed to enrich '%s' (%s): %s", token.lemma, token.pos, e)
        return token, None


async def _process_file(file_path: Path, skip_enrichment: bool) -> None:
    """Async implementation of process command."""
    console.print(f"\n[bold]Processing:[/] {file_path.name}\n")

    try:
        # Step 1: Extract text
        with create_simple_progress() as progress:
            task = progress.add_task("Extracting text...")
            extractor = TextExtractor(whisper_model=settings.whisper_model)
            raw_text = await extractor.extract(file_path)
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
            return

        # Step 3: Identify new vs duplicate tokens
        console.print(f"\n[info]Processing {len(tokens)} vocabulary items...[/]\n")

        async with async_session() as session:
            new_tokens: list[TokenInfo] = []
            duplicates = 0

            with create_simple_progress() as progress:
                task = progress.add_task("Checking for duplicates...")
                for token in tokens:
                    existing_word = await _find_existing_word(session, token.lemma, token.pos)
                    if existing_word:
                        duplicates += 1
                    else:
                        new_tokens.append(token)
                progress.update(task, description=f"[green]Found {len(new_tokens)} new words[/]")

            # Step 4: Parallel enrichment for new tokens
            enricher = Enricher() if not skip_enrichment else None
            enrichments: list[tuple[TokenInfo, EnrichmentResult | None]] = []
            errors = 0

            if new_tokens and enricher:
                with create_progress() as progress:
                    task = progress.add_task("Enriching new words...", total=len(new_tokens))

                    # Run all enrichments in parallel (semaphore in llm.py limits concurrency)
                    enrichments = await asyncio.gather(
                        *[_enrich_token(enricher, t) for t in new_tokens]
                    )

                    # Count errors
                    for token, enrichment in enrichments:
                        if enrichment is None:
                            errors += 1
                        elif hasattr(enrichment, "error") and enrichment.error:
                            errors += 1
                            logger.warning(
                                "Enrichment error for '%s': %s", token.lemma, enrichment.error
                            )
                        progress.update(task, advance=1)
            elif new_tokens:
                # Skip enrichment - just pair tokens with None
                enrichments = [(t, None) for t in new_tokens]

            # Step 5: Create Word records for new tokens
            new_words_created = 0
            if enrichments:
                with create_simple_progress() as progress:
                    task = progress.add_task("Creating word records...")
                    for token, enrichment in enrichments:
                        word = Word(
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
                        )
                        session.add(word)
                        new_words_created += 1

                    await session.commit()
                    progress.update(
                        task, description=f"[green]Created {new_words_created} words[/]"
                    )

        console.print()
        console.print("[success]Processing complete![/]")
        console.print(f"  New words: [green]{new_words_created}[/]")
        console.print(f"  Duplicates: [dim]{duplicates}[/]")
        if errors:
            console.print(f"  Enrichment errors: [yellow]{errors}[/]")
        console.print()
        console.print("[info]Run 'vocabext sync' to sync new words to Anki.[/]")

    except Exception as e:
        error_console.print(f"[error]Processing failed: {e}[/]")
        raise typer.Exit(1) from None


async def _find_existing_word(session: "AsyncSession", lemma: str, pos: str) -> Word | None:
    """Find an existing word in vocabulary."""
    stmt = select(Word).where(Word.lemma == lemma, Word.pos == pos)
    result = await session.execute(stmt)
    word: Word | None = result.scalar_one_or_none()
    return word
