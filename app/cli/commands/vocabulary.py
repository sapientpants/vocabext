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
from app.services.enricher import Enricher, EnrichmentResult, filter_non_german_words
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
    exact: bool = False,
) -> Select[tuple[Word]]:
    """Build word query with filters.

    Args:
        search: Search term for lemma matching
        exact: If True, match lemma exactly. If False (default), match lemmas starting with search term.
    """
    stmt = select(Word)
    if random_order:
        stmt = stmt.order_by(func.random())
    else:
        stmt = stmt.order_by(Word.lemma)

    if search:
        if exact:
            stmt = stmt.where(Word.lemma.ilike(search))
        else:
            escaped_search = _escape_like_pattern(search)
            stmt = stmt.where(Word.lemma.ilike(f"{escaped_search}%"))
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
        "cases": word.cases,
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
    if enrichment.cases:
        word.cases = json.dumps(enrichment.cases)
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
        "cases": word.cases,
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
            cases=old_values["cases"],
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
    search: str = typer.Option("", "--search", "-s", help="Search by lemma (prefix match)"),
    pos: str = typer.Option("", "--pos", "-p", help="Filter by POS (NOUN, VERB, ADJ, ADV, ADP)"),
    unsynced: bool = typer.Option(False, "--unsynced", "-u", help="Show only unsynced words"),
    needs_review: bool = typer.Option(
        False, "--review", "-r", help="Show only words needing review"
    ),
    exact: bool = typer.Option(False, "--exact", "-e", help="Match lemma exactly"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of words to show"),
) -> None:
    """List vocabulary words with optional filters."""
    run_async(_list_words(search, pos, unsynced, needs_review, exact, limit))


async def _list_words(
    search: str, pos: str, unsynced: bool, needs_review: bool, exact: bool, limit: int
) -> None:
    """Async implementation of list command."""
    async with async_session() as session:
        sync_status = "unsynced" if unsynced else ""
        review_status = "needs_review" if needs_review else ""
        stmt = build_filtered_query(
            search, pos, sync_status, review_status=review_status, exact=exact
        )
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


def _is_valid_german_word(word: str) -> tuple[bool, str | None]:
    """Check if word passes basic validation rules.

    Returns:
        Tuple of (is_valid, error_reason). If valid, error_reason is None.
    """
    # Must be at least 2 characters (same as tokenize)
    if len(word) < 2:
        return False, "too short (minimum 2 characters)"

    # Must contain only alphabetic characters (including German umlauts and ß)
    if not word.isalpha():
        return False, "contains non-alphabetic characters"

    return True, None


async def _add_word(word: str, context: str) -> None:
    """Async implementation of add command."""
    word = word.strip()
    if not word:
        error_console.print("[error]Word cannot be empty[/]")
        raise typer.Exit(1)

    # Validate word passes basic checks
    is_valid, error_reason = _is_valid_german_word(word)
    if not is_valid:
        error_console.print(f"[error]Invalid word: {error_reason}[/]")
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
                    # analyze_word returns None for proper nouns (names, places, etc.)
                    raise ValueError(
                        f"'{word}' appears to be a proper noun (name/place) and cannot be added"
                    )

                pos = token_info.pos
                lemma = token_info.lemma
                progress.update(task, description=f"[dim]Detected: {pos}[/] Enriching...")

                # Dictionary validation (spaCy) + single LLM call for enrichment
                # Same pipeline as process file - see docs/llm-pipelines.md
                enrichment = await enricher.enrich_with_dictionary(lemma, pos, context, session)
                progress.update(task, description=f"[green]Complete: {pos}[/]")
            except Exception as e:
                progress.update(task, description="[red]Failed[/]")
                logger.exception("Failed to enrich word")
                error_console.print(f"[error]Failed to enrich word: {e}[/]")
                raise typer.Exit(1) from e

        if not enrichment:
            error_console.print("[error]Enrichment returned no result[/]")
            raise typer.Exit(1)

        # Check for duplicate
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

        # Create Word record
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
            cases=json.dumps(enrichment.cases) if enrichment.cases else None,
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
    search: str = typer.Option("", "--search", "-s", help="Filter by lemma (prefix match)"),
    pos: str = typer.Option("", "--pos", "-p", help="Filter by POS"),
    all_words: bool = typer.Option(False, "--all", "-a", help="Validate all words"),
    exact: bool = typer.Option(False, "--exact", "-e", help="Match lemma exactly"),
    limit: int = typer.Option(0, "--limit", "-n", help="Maximum words to validate (0 = no limit)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would change without applying"
    ),
) -> None:
    """Batch validate words with LLM enrichment."""
    if not all_words and not search and not pos:
        error_console.print("[error]Specify --search, --pos, or use --all to validate all words[/]")
        raise typer.Exit(1)

    run_async(_validate_words(search, pos, exact, limit, dry_run))


async def _validate_words(
    search: str, pos: str, exact: bool, limit: int, dry_run: bool = False
) -> None:
    """Async implementation of validate command."""
    # Pre-load spaCy model before starting (takes ~30-60s on first run)
    if not is_model_loaded():
        console.print("[dim]Loading language model...[/]", end="", highlight=False)
        preload_model()
        console.print(" [green]done[/]")

    enricher = Enricher()

    async with async_session() as session:
        stmt = build_filtered_query(search, pos, random_order=True, exact=exact)
        if limit > 0:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        words = list(result.scalars().all())

        if not words:
            console.print("[dim]No words found matching filters.[/]")
            return

        mode_label = "[yellow](DRY RUN)[/] " if dry_run else ""
        console.print(f"[info]{mode_label}Validating {len(words)} words...[/]\n")

        deleted = 0
        # Track details for dry-run summary
        would_delete: list[tuple[Word, str]] = []

        def validate_word_basic(word: Word) -> str | None:
            """Check basic validation rules. Returns error reason or None if valid."""
            is_valid, error_reason = _is_valid_german_word(word.lemma)
            return error_reason if not is_valid else None

        # Step 1: Basic validation (alphabetic check)
        valid_words: list[Word] = []
        with create_simple_progress() as progress:
            task_id = progress.add_task("Checking format...", total=len(words))
            for word in words:
                validation_error = validate_word_basic(word)
                if validation_error:
                    if dry_run:
                        would_delete.append((word, validation_error))
                    else:
                        await record_event(
                            session,
                            word,
                            "DELETED",
                            "validate",
                            f"Validation failed: {validation_error}",
                        )
                        await session.delete(word)
                    deleted += 1
                else:
                    valid_words.append(word)
                progress.advance(task_id)

        # Commit alphabetic check deletions
        if not dry_run and deleted > 0:
            await session.commit()

        # Step 2: LLM-based language check
        if valid_words:
            lemmas = [w.lemma for w in valid_words]
            num_batches = (len(lemmas) + 49) // 50  # Ceiling division

            with create_progress() as progress:
                task_id = progress.add_task("Checking for non-German words...", total=num_batches)

                def on_progress(completed: int, total: int) -> None:
                    progress.update(task_id, completed=completed)

                non_german_lemmas = await filter_non_german_words(lemmas, on_progress)

            if non_german_lemmas:
                # Filter out non-German words
                remaining_valid: list[Word] = []
                for word in valid_words:
                    if word.lemma in non_german_lemmas:
                        if dry_run:
                            would_delete.append((word, "not a German word"))
                        else:
                            await record_event(
                                session,
                                word,
                                "DELETED",
                                "validate",
                                "Validation failed: not a German word",
                            )
                            await session.delete(word)
                        deleted += 1
                    else:
                        remaining_valid.append(word)
                valid_words = remaining_valid

                # Commit LLM check deletions
                if not dry_run:
                    await session.commit()

        # Show deletion summary
        if not dry_run and deleted > 0:
            console.print(f"[dim]Deleted {deleted} invalid words[/]\n")

        # If no valid words remain or dry-run, we're done after validation
        if not valid_words or dry_run:
            if dry_run:
                console.print(
                    f"[yellow]DRY RUN[/] Would delete: {deleted}, "
                    f"Would enrich: {len(valid_words)} (skipped)"
                )
                if would_delete:
                    console.print("\n[bold]Would delete:[/]")
                    for word, reason in would_delete[:10]:
                        console.print(f"  - {word.display_word} ({word.id}): {reason}")
                    if len(would_delete) > 10:
                        console.print(f"  ... and {len(would_delete) - 10} more")
            else:
                console.print(f"[success]Complete![/] Deleted: {deleted}, No valid words to enrich")
            return

        # Step 3: Enrich remaining valid words (only if not dry-run)
        modified, skipped, enrichment_deleted, errors = 0, 0, 0, 0

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
        tasks = [asyncio.create_task(enrich_word(word)) for word in valid_words]

        def status_description() -> str:
            """Build a fixed-width status description."""
            total_deleted = deleted + enrichment_deleted
            return f"[green]M:{modified}[/] [dim]S:{skipped}[/] [yellow]D:{total_deleted}[/] [red]E:{errors}[/]"

        with create_progress() as progress:
            task_id = progress.add_task("Enriching...", total=len(valid_words))

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
                    try:
                        status = await apply_enrichment_to_word(word, enrich_result, session)
                        if status == "modified":
                            modified += 1
                        elif status == "deleted":
                            enrichment_deleted += 1
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
                await session.commit()

        console.print()
        total_deleted = deleted + enrichment_deleted
        console.print(
            f"[success]Complete![/] Modified: {modified}, Skipped: {skipped}, "
            f"Deleted: {total_deleted}, Errors: {errors}"
        )


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
    word_id_or_lemma: str = typer.Argument(..., help="Word ID or lemma to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a vocabulary word by ID or lemma (exact match)."""
    run_async(_delete_word(word_id_or_lemma, force))


async def _delete_word(word_id_or_lemma: str, force: bool) -> None:
    """Async implementation of delete command."""
    async with async_session() as session:
        # Check if input is numeric (ID) or string (lemma)
        if word_id_or_lemma.isdigit():
            word = await session.get(Word, int(word_id_or_lemma))
            if not word:
                error_console.print(f"[error]Word with ID {word_id_or_lemma} not found[/]")
                raise typer.Exit(1)
        else:
            # Exact search by lemma
            stmt = select(Word).where(Word.lemma.ilike(word_id_or_lemma))
            result = await session.execute(stmt)
            words = result.scalars().all()

            if not words:
                error_console.print(f"[error]No word found with lemma '{word_id_or_lemma}'[/]")
                raise typer.Exit(1)

            if len(words) > 1:
                # Multiple matches (e.g., same lemma with different POS/gender)
                error_console.print(
                    f"[error]Multiple words match '{word_id_or_lemma}'. Use ID instead:[/]"
                )
                for w in words:
                    console.print(f"  ID {w.id}: {w.display_word} ({w.pos})")
                raise typer.Exit(1)

            word = words[0]

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
