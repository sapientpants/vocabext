"""Vocabulary management commands."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone

import typer
from rich.prompt import Confirm, Prompt
from rich.table import Table
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.cli.utils.async_runner import run_async
from app.cli.utils.console import console, error_console
from app.cli.utils.progress import create_progress, create_simple_progress
from app.database import async_session
from app.models import Word, WordEvent, WordVersion
from app.services.dictionary import is_model_loaded, preload_model
from app.services.enricher import Enricher, EnrichmentResult
from app.services.events import (
    get_deleted_words,
    get_word_history,
    record_event,
    revert_to_event,
    undo_last_change,
)
from app.services.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _escape_like_pattern(s: str) -> str:
    """Escape special characters for SQL LIKE patterns."""
    # Escape %, _, and \ which are special in LIKE
    return re.sub(r"([%_\\])", r"\\\1", s)


app = typer.Typer(
    name="vocab",
    help="Vocabulary management commands",
    no_args_is_help=True,
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
    """Build word query with filters."""
    stmt = select(Word)
    if random_order:
        stmt = stmt.order_by(func.random())
    else:
        stmt = stmt.order_by(Word.lemma)

    if search:
        escaped_search = _escape_like_pattern(search)
        stmt = stmt.where(Word.lemma.ilike(f"%{escaped_search}%"))
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


async def version_exists(session: AsyncSession, word_id: int, version_number: int) -> bool:
    """Check if a version with the given number already exists for this word."""
    stmt = select(WordVersion).where(
        WordVersion.word_id == word_id,
        WordVersion.version_number == version_number,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def apply_enrichment_to_word(
    word: Word, enrichment: EnrichmentResult, session: AsyncSession
) -> str:
    """
    Apply LLM enrichment to word, creating version and event if changed.

    Returns: 'modified', 'deleted', or 'skipped'
    """
    # Check for duplicate lemma - DELETE the word if it would create a duplicate
    if enrichment.lemma and enrichment.lemma != word.lemma:
        if await check_duplicate_lemma(session, enrichment.lemma, word.pos, word.id):
            # Record deletion event BEFORE deleting (captures current state)
            await record_event(
                session,
                word,
                "DELETED",
                "validate",
                f"Duplicate of '{enrichment.lemma}' after lemma correction",
            )
            await session.delete(word)
            return "deleted"

    # Capture old values
    old_values = {
        "lemma": word.lemma,
        "gender": word.gender,
        "plural": word.plural,
        "preterite": word.preterite,
        "past_participle": word.past_participle,
        "auxiliary": word.auxiliary,
        "translations": word.translations,
        "lemma_source": word.lemma_source,
        "frequency": word.frequency,
        "ipa": word.ipa,
        "dictionary_url": word.dictionary_url,
        "definition_de": word.definition_de,
        "synonyms": word.synonyms,
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

    # Apply dictionary-grounded fields (important for idempotency)
    if enrichment.lemma_source:
        word.lemma_source = enrichment.lemma_source
    if enrichment.frequency is not None:
        word.frequency = enrichment.frequency
    if enrichment.ipa:
        word.ipa = enrichment.ipa
    if enrichment.dictionary_url:
        word.dictionary_url = enrichment.dictionary_url
    if enrichment.definition_de:
        word.definition_de = enrichment.definition_de
    if enrichment.synonyms:
        word.synonyms = json.dumps(enrichment.synonyms)

    # Check if anything changed
    new_values = {
        "lemma": word.lemma,
        "gender": word.gender,
        "plural": word.plural,
        "preterite": word.preterite,
        "past_participle": word.past_participle,
        "auxiliary": word.auxiliary,
        "translations": word.translations,
        "lemma_source": word.lemma_source,
        "frequency": word.frequency,
        "ipa": word.ipa,
        "dictionary_url": word.dictionary_url,
        "definition_de": word.definition_de,
        "synonyms": word.synonyms,
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
    word.needs_review = False
    word.review_reason = None

    # Record modification event with new state
    await record_event(session, word, "MODIFIED", "validate", "Enrichment applied")

    return "modified"


@app.command(name="list")
def list_words(
    search: str = typer.Option("", "--search", "-s", help="Search by lemma"),
    pos: str = typer.Option("", "--pos", "-p", help="Filter by POS (NOUN, VERB, ADJ, ADV, ADP)"),
    unsynced: bool = typer.Option(False, "--unsynced", "-u", help="Show only unsynced words"),
    needs_review: bool = typer.Option(
        False, "--review", "-r", help="Show only words needing review"
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of words to show"),
) -> None:
    """List vocabulary words with optional filters."""
    run_async(_list_words(search, pos, unsynced, needs_review, limit))


async def _list_words(
    search: str, pos: str, unsynced: bool, needs_review: bool, limit: int
) -> None:
    """Async implementation of list command."""
    async with async_session() as session:
        sync_status = "unsynced" if unsynced else ""
        review_status = "needs_review" if needs_review else ""
        stmt = build_filtered_query(search, pos, sync_status, review_status=review_status)
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        words = result.scalars().all()

        if not words:
            console.print("[dim]No words found matching filters.[/]")
            return

        # Build Rich table
        table = Table(title=f"Vocabulary ({len(words)} words)")
        table.add_column("ID", style="dim", justify="right")
        table.add_column("Word", style="word")
        table.add_column("POS", style="pos")
        table.add_column("Grammar")
        table.add_column("Translations")
        table.add_column("Synced", justify="center")

        for word in words:
            translations = word.translations_list[:2]
            trans_str = ", ".join(translations)
            if len(word.translations_list) > 2:
                trans_str += "..."

            synced_str = "[green]Yes[/]" if word.is_synced else "[yellow]No[/]"
            if word.needs_review:
                synced_str = "[red]Review[/]"

            table.add_row(
                str(word.id),
                word.display_word,
                word.pos,
                word.grammar_info or "",
                trans_str,
                synced_str,
            )

        console.print(table)


@app.command(name="add")
def add_word(
    word: str = typer.Argument(..., help="German word to add"),
    context: str = typer.Option(
        "", "--context", "-c", help="Context sentence for better POS detection"
    ),
) -> None:
    """Add a single word to the vocabulary.

    Automatically detects the part of speech and enriches the word
    with grammatical information, just like words from uploaded files.
    """
    run_async(_add_word(word, context))


def _is_valid_german_word(word: str) -> bool:
    """Check if word contains only alphabetic characters (including German umlauts and ß)."""
    # Python's str.isalpha() correctly handles German characters (äöüß etc.)
    return word.isalpha()


async def _add_word(word: str, context: str) -> None:
    """Async implementation of add command."""
    word = word.strip()
    if not word:
        error_console.print("[error]Word cannot be empty[/]")
        raise typer.Exit(1)

    # Validate word contains only alphabetic characters
    if not _is_valid_german_word(word):
        error_console.print(
            "[error]Word must contain only alphabetic characters (including German umlauts)[/]"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold]Adding word:[/] {word}\n")

    # Pre-load spaCy model if needed
    if not is_model_loaded():
        console.print("[dim]Loading language model...[/]", end="", highlight=False)
        preload_model()
        console.print(" [green]done[/]")

    async with async_session() as session:
        # Step 1: Use spaCy to detect POS and normalize lemma (same as process file)
        tokenizer = Tokenizer()
        enricher = Enricher()
        pos: str | None = None
        enrichment: EnrichmentResult | None = None

        with create_simple_progress() as progress:
            task = progress.add_task("Analyzing word...")

            try:
                # Use spaCy for POS detection (same pipeline as document processing)
                token_info = tokenizer.analyze_word(word, context)
                if token_info is None:
                    raise ValueError(f"Could not analyze word: {word}")

                pos = token_info.pos
                lemma = token_info.lemma
                progress.update(task, description=f"[dim]Detected: {pos}[/] Enriching...")

                # Single LLM call for enrichment (translations + grammar)
                enrichment = await enricher.enrich(lemma, pos, token_info.context_sentence)
                progress.update(task, description=f"[green]Complete: {pos}[/]")
            except Exception as e:
                progress.update(task, description="[red]Failed[/]")
                logger.exception("Failed to enrich word")
                error_console.print(f"[error]Failed to enrich word: {e}[/]")
                raise typer.Exit(1) from e

        if not enrichment:
            error_console.print("[error]Enrichment returned no result[/]")
            raise typer.Exit(1)

        # Use the lemma from spaCy analysis (enrichment.lemma is LLM's version, not authoritative)
        # This matches the process file pipeline where spaCy determines the lemma

        # Step 2: Check for duplicate
        # Include gender for nouns to align with database unique constraint on (lemma, pos, gender)
        conditions = [Word.lemma == lemma, Word.pos == pos]
        if pos == "NOUN" and enrichment.gender is not None:
            conditions.append(Word.gender == enrichment.gender)

        stmt = select(Word).where(*conditions)
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            error_console.print(
                f"[warning]Word already exists: {existing.display_word} (ID: {existing.id})[/]"
            )
            if existing.translations_list:
                console.print(f"[dim]Translations: {', '.join(existing.translations_list)}[/]")
            raise typer.Exit(1)

        # Step 3: Create Word record
        needs_review = bool(enrichment.error)
        review_reason = enrichment.error if enrichment.error else None

        new_word = Word(
            lemma=lemma,
            pos=pos,
            gender=enrichment.gender,
            plural=enrichment.plural,
            preterite=enrichment.preterite,
            past_participle=enrichment.past_participle,
            auxiliary=enrichment.auxiliary,
            translations=json.dumps(enrichment.translations) if enrichment.translations else None,
            definition_de=enrichment.definition_de,
            synonyms=json.dumps(enrichment.synonyms) if enrichment.synonyms else None,
            frequency=enrichment.frequency,
            ipa=enrichment.ipa,
            lemma_source="spacy",  # Lemma comes from spaCy tokenizer
            dictionary_url=enrichment.dictionary_url,
            needs_review=needs_review,
            review_reason=review_reason,
        )
        session.add(new_word)
        await session.flush()

        # Step 4: Record creation event
        await record_event(
            session,
            new_word,
            "CREATED",
            source="cli",
            reason="Added via 'vocab add'",
        )

        await session.commit()

        # Step 5: Display result
        console.print()
        console.print(f"[success]Added word: {new_word.display_word}[/]")
        console.print(f"  ID: [dim]{new_word.id}[/]")
        console.print(f"  POS: [pos]{new_word.pos}[/]")

        if new_word.grammar_info:
            console.print(f"  Grammar: {new_word.grammar_info}")

        if new_word.translations_list:
            console.print(f"  Translations: {', '.join(new_word.translations_list)}")

        if new_word.ipa:
            console.print(f"  IPA: {new_word.ipa}")

        if new_word.frequency is not None:
            console.print(f"  Frequency: {new_word.frequency:.1f}/6")

        if new_word.needs_review:
            console.print(f"  [warning]Needs review: {new_word.review_reason}[/]")

        console.print()
        console.print("[info]Run 'vocabext sync' to sync to Anki.[/]")


@app.command(name="validate")
def validate_words(
    search: str = typer.Option("", "--search", "-s", help="Filter by lemma search"),
    pos: str = typer.Option("", "--pos", "-p", help="Filter by POS"),
    all_words: bool = typer.Option(False, "--all", "-a", help="Validate all words"),
    limit: int = typer.Option(0, "--limit", "-n", help="Maximum words to validate (0 = no limit)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would change without applying"
    ),
) -> None:
    """Batch validate words with LLM enrichment."""
    if not all_words and not search and not pos:
        error_console.print("[error]Specify --search, --pos, or use --all to validate all words[/]")
        raise typer.Exit(1)

    run_async(_validate_words(search, pos, limit, dry_run))


async def _validate_words(search: str, pos: str, limit: int, dry_run: bool = False) -> None:
    """Async implementation of validate command."""
    # Pre-load spaCy model before starting (takes ~30-60s on first run)
    if not is_model_loaded():
        console.print("[dim]Loading language model...[/]", end="", highlight=False)
        preload_model()
        console.print(" [green]done[/]")

    enricher = Enricher()

    async with async_session() as session:
        stmt = build_filtered_query(search, pos, random_order=True)
        if limit > 0:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        words = list(result.scalars().all())

        if not words:
            console.print("[dim]No words found matching filters.[/]")
            return

        mode_label = "[yellow](DRY RUN)[/] " if dry_run else ""
        console.print(f"[info]{mode_label}Validating {len(words)} words...[/]\n")

        modified, skipped, deleted, errors = 0, 0, 0, 0
        # Track details for dry-run summary
        would_delete: list[tuple[Word, str]] = []
        would_modify: list[tuple[Word, EnrichmentResult]] = []

        async def enrich_word(word: Word) -> tuple[Word, EnrichmentResult | Exception]:
            """Enrich a single word, return (word, result) tuple."""
            try:
                # Use dictionary-based enrichment for more stable lemma validation.
                # Note: session is not passed to avoid concurrent cache write conflicts.
                # The spaCy vocabulary is used for validation (in-memory, no DB needed).
                # Dictionary cache is populated on single-word operations (add, edit).
                enrichment = await enricher.enrich_with_dictionary(word.lemma, word.pos, context="")
                return (word, enrichment)
            except Exception as e:
                return (word, e)

        # Fire all tasks - semaphore in llm.chat_completion limits concurrency
        tasks = [asyncio.create_task(enrich_word(word)) for word in words]

        def status_description() -> str:
            """Build a fixed-width status description."""
            return f"[green]M:{modified}[/] [dim]S:{skipped}[/] [yellow]D:{deleted}[/] [red]E:{errors}[/]"

        with create_progress() as progress:
            task_id = progress.add_task("Validating...", total=len(words))

            for coro in asyncio.as_completed(tasks):
                word, enrich_result = await coro

                if isinstance(enrich_result, Exception):
                    errors += 1
                    logger.error(
                        "Failed to enrich '%s' (%s): %s",
                        word.display_word,
                        word.pos,
                        enrich_result,
                        exc_info=False,
                    )
                elif enrich_result.error:
                    errors += 1
                    logger.warning(
                        "Enrichment error for '%s': %s", word.display_word, enrich_result.error
                    )
                else:
                    if dry_run:
                        # Simulate what would happen without applying
                        status = await _simulate_enrichment(word, enrich_result, session)
                        if status == "modified":
                            modified += 1
                            would_modify.append((word, enrich_result))
                        elif status == "deleted":
                            deleted += 1
                            reason = f"Duplicate of '{enrich_result.lemma}'"
                            would_delete.append((word, reason))
                        else:
                            skipped += 1
                    else:
                        try:
                            status = await apply_enrichment_to_word(word, enrich_result, session)
                            if status == "modified":
                                modified += 1
                            elif status == "deleted":
                                deleted += 1
                            else:
                                skipped += 1
                        except Exception as e:
                            errors += 1
                            logger.error(
                                "Failed to apply enrichment to '%s': %s",
                                word.display_word,
                                e,
                                exc_info=True,
                            )

                progress.update(task_id, advance=1, description=status_description())

                # Commit after each word (only if not dry run)
                if not dry_run:
                    await session.commit()

        console.print()

        if dry_run:
            console.print(
                f"[yellow]DRY RUN[/] Would modify: {modified}, Would delete: {deleted}, "
                f"Unchanged: {skipped}, Errors: {errors}"
            )
            if would_delete:
                console.print("\n[bold]Would delete:[/]")
                for word, reason in would_delete[:10]:
                    console.print(f"  - {word.display_word} ({word.id}): {reason}")
                if len(would_delete) > 10:
                    console.print(f"  ... and {len(would_delete) - 10} more")
            if would_modify:
                console.print("\n[bold]Would modify:[/]")
                for word, enrichment in would_modify[:10]:
                    changes = []
                    if enrichment.lemma and enrichment.lemma != word.lemma:
                        changes.append(f"lemma: {word.lemma} → {enrichment.lemma}")
                    if enrichment.gender and enrichment.gender != word.gender:
                        changes.append(f"gender: {word.gender} → {enrichment.gender}")
                    if enrichment.plural and enrichment.plural != word.plural:
                        changes.append(f"plural: {word.plural} → {enrichment.plural}")
                    console.print(
                        f"  - {word.display_word} ({word.id}): {', '.join(changes) or 'translations updated'}"
                    )
                if len(would_modify) > 10:
                    console.print(f"  ... and {len(would_modify) - 10} more")
        else:
            console.print(
                f"[success]Complete![/] Modified: {modified}, Skipped: {skipped}, "
                f"Deleted: {deleted}, Errors: {errors}"
            )


async def _simulate_enrichment(
    word: Word, enrichment: EnrichmentResult, session: AsyncSession
) -> str:
    """
    Simulate what would happen if enrichment were applied, without modifying anything.

    Returns: 'modified', 'deleted', or 'skipped'
    """
    # Check for duplicate lemma - would DELETE the word
    if enrichment.lemma and enrichment.lemma != word.lemma:
        if await check_duplicate_lemma(session, enrichment.lemma, word.pos, word.id):
            return "deleted"

    # Check if anything would change
    would_change = (
        (enrichment.lemma and enrichment.lemma != word.lemma)
        or (enrichment.gender and enrichment.gender != word.gender)
        or (enrichment.plural and enrichment.plural != word.plural)
        or (enrichment.preterite and enrichment.preterite != word.preterite)
        or (enrichment.past_participle and enrichment.past_participle != word.past_participle)
        or (enrichment.auxiliary and enrichment.auxiliary != word.auxiliary)
        or (enrichment.translations and json.dumps(enrichment.translations) != word.translations)
    )

    return "modified" if would_change else "skipped"


@app.command(name="edit")
def edit_word(
    word_id: int = typer.Argument(..., help="Word ID to edit"),
) -> None:
    """Edit a vocabulary word interactively."""
    run_async(_edit_word(word_id))


async def _edit_word(word_id: int) -> None:
    """Async implementation of edit command."""
    async with async_session() as session:
        stmt = select(Word).where(Word.id == word_id).options(selectinload(Word.versions))
        result = await session.execute(stmt)
        word = result.scalar_one_or_none()

        if not word:
            error_console.print(f"[error]Word with ID {word_id} not found[/]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Editing word:[/] {word.display_word} ({word.pos})")
        console.print(f"[dim]Current grammar:[/] {word.grammar_info or 'None'}")
        console.print(
            f"[dim]Current translations:[/] {', '.join(word.translations_list) or 'None'}\n"
        )

        # Collect new values
        new_lemma = Prompt.ask("Lemma", default=word.lemma)

        new_gender = None
        new_plural = None
        new_preterite = None
        new_past_participle = None
        new_auxiliary = None

        if word.pos == "NOUN":
            new_gender = Prompt.ask("Gender (der/die/das)", default=word.gender or "")
            new_plural = Prompt.ask("Plural", default=word.plural or "")
        elif word.pos == "VERB":
            new_preterite = Prompt.ask("Preterite", default=word.preterite or "")
            new_past_participle = Prompt.ask("Past participle", default=word.past_participle or "")
            new_auxiliary = Prompt.ask("Auxiliary (haben/sein)", default=word.auxiliary or "haben")

        new_translations = Prompt.ask(
            "Translations (comma-separated)",
            default=", ".join(word.translations_list),
        )

        # Check for changes
        trans_list = [t.strip() for t in new_translations.split(",") if t.strip()]
        translations_json = json.dumps(trans_list) if trans_list else None

        has_changes = (
            new_lemma != word.lemma
            or new_gender != word.gender
            or new_plural != word.plural
            or new_preterite != word.preterite
            or new_past_participle != word.past_participle
            or new_auxiliary != word.auxiliary
            or translations_json != word.translations
        )

        if not has_changes:
            console.print("[dim]No changes made.[/]")
            return

        # Create version snapshot
        if not await version_exists(session, word.id, word.current_version):
            version = WordVersion(
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
            session.add(version)

        # Apply changes
        word.lemma = new_lemma
        word.gender = new_gender or None
        word.plural = new_plural or None
        word.preterite = new_preterite or None
        word.past_participle = new_past_participle or None
        word.auxiliary = new_auxiliary or None
        word.translations = translations_json
        word.current_version += 1
        word.anki_synced_at = None

        await session.commit()
        console.print("[success]Word updated successfully![/]")


@app.command(name="delete")
def delete_word(
    word_id: int = typer.Argument(..., help="Word ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a vocabulary word."""
    run_async(_delete_word(word_id, force))


async def _delete_word(word_id: int, force: bool) -> None:
    """Async implementation of delete command."""
    async with async_session() as session:
        word = await session.get(Word, word_id)

        if not word:
            error_console.print(f"[error]Word with ID {word_id} not found[/]")
            raise typer.Exit(1)

        if not force:
            console.print(f"\n[bold]Word to delete:[/] {word.display_word} ({word.pos})")
            console.print(f"[dim]Translations:[/] {', '.join(word.translations_list) or 'None'}")

            if not Confirm.ask("\nAre you sure you want to delete this word?"):
                console.print("[dim]Cancelled.[/]")
                return

        # Record deletion event before deleting
        await record_event(session, word, "DELETED", "cli", "Manual deletion")
        await session.delete(word)
        await session.commit()
        console.print(f"[success]Deleted word: {word.display_word}[/]")


@app.command(name="history")
def show_history(
    word_id: int = typer.Argument(..., help="Word ID to show history for"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum events to show"),
) -> None:
    """Show change history for a word."""
    run_async(_show_history(word_id, limit))


async def _show_history(word_id: int, limit: int) -> None:
    """Async implementation of history command."""
    async with async_session() as session:
        events = await get_word_history(session, word_id, limit)

        if not events:
            # Check if word exists
            word = await session.get(Word, word_id)
            if word:
                console.print(
                    f"[dim]No history recorded for word {word_id} ({word.display_word})[/]"
                )
            else:
                error_console.print(f"[error]Word with ID {word_id} not found[/]")
            return

        table = Table(title=f"History for word {word_id}")
        table.add_column("Event ID", style="dim")
        table.add_column("Time")
        table.add_column("Type")
        table.add_column("Word")
        table.add_column("Source")
        table.add_column("Reason")

        for event in events:
            type_style = {
                "CREATED": "green",
                "MODIFIED": "yellow",
                "DELETED": "red",
                "RESTORED": "cyan",
            }.get(event.event_type, "white")

            display_word = event.lemma
            if event.pos == "NOUN" and event.gender:
                display_word = f"{event.gender} {event.lemma}"

            table.add_row(
                str(event.id),
                event.event_at.strftime("%Y-%m-%d %H:%M"),
                f"[{type_style}]{event.event_type}[/]",
                display_word,
                event.source,
                event.reason or "-",
            )

        console.print(table)


@app.command(name="deleted")
def list_deleted(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum to show"),
    hours: int = typer.Option(24, "--hours", "-h", help="Show deletions from last N hours"),
) -> None:
    """List recently deleted words that can be restored."""
    run_async(_list_deleted(limit, hours))


async def _list_deleted(limit: int, hours: int) -> None:
    """Async implementation of deleted command."""
    async with async_session() as session:
        after = datetime.now(timezone.utc) - timedelta(hours=hours)
        events = await get_deleted_words(session, after=after, limit=limit)

        if not events:
            console.print(f"[dim]No deleted words in the last {hours} hours[/]")
            return

        table = Table(title=f"Deleted Words (last {hours}h)")
        table.add_column("Event ID", style="dim")
        table.add_column("Word ID")
        table.add_column("Word")
        table.add_column("Deleted At")
        table.add_column("Reason")

        for event in events:
            display_word = event.lemma
            if event.pos == "NOUN" and event.gender:
                display_word = f"{event.gender} {event.lemma}"

            table.add_row(
                str(event.id),
                str(event.word_id),
                display_word,
                event.event_at.strftime("%Y-%m-%d %H:%M"),
                event.reason or "-",
            )

        console.print(table)
        console.print("\n[dim]Use 'vocabext vocab restore <event_id>' to restore a word[/]")


@app.command(name="restore")
def restore_word(
    event_id: int = typer.Argument(..., help="Event ID to restore from"),
) -> None:
    """Restore a deleted word from an event."""
    run_async(_restore_word(event_id))


async def _restore_word(event_id: int) -> None:
    """Async implementation of restore command."""
    async with async_session() as session:
        event = await session.get(WordEvent, event_id)

        if not event:
            error_console.print(f"[error]Event with ID {event_id} not found[/]")
            raise typer.Exit(1)

        # Check if word already exists
        existing = await session.get(Word, event.word_id)
        if existing:
            console.print(
                f"[warning]Word {event.word_id} already exists ({existing.display_word})[/]"
            )
            if not Confirm.ask("Overwrite with state from this event?"):
                console.print("[dim]Cancelled.[/]")
                return

        word = await revert_to_event(session, event, source="restore")
        await session.commit()

        console.print(f"[success]Restored word: {word.display_word}[/]")


@app.command(name="undo")
def undo_word(
    word_id: int = typer.Argument(..., help="Word ID to undo last change"),
) -> None:
    """Undo the last change to a word."""
    run_async(_undo_word(word_id))


async def _undo_word(word_id: int) -> None:
    """Async implementation of undo command."""
    async with async_session() as session:
        word = await undo_last_change(session, word_id)

        if not word:
            error_console.print(f"[error]Cannot undo - no previous state for word {word_id}[/]")
            raise typer.Exit(1)

        await session.commit()
        console.print(f"[success]Undid last change to: {word.display_word}[/]")
