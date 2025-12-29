"""Vocabulary management commands."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import typer
from rich.prompt import Confirm, Prompt
from rich.table import Table
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.cli.utils.async_runner import run_async
from app.cli.utils.console import console, error_console
from app.cli.utils.progress import create_progress
from app.database import async_session
from app.models import Word, WordVersion
from app.services.enricher import Enricher, EnrichmentResult

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
    Apply LLM enrichment to word, creating version if changed.

    Returns: 'modified', 'deleted', or 'skipped'
    """
    # Check for duplicate lemma - DELETE the word if it would create a duplicate
    if enrichment.lemma and enrichment.lemma != word.lemma:
        if await check_duplicate_lemma(session, enrichment.lemma, word.pos, word.id):
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
    word.needs_review = False
    word.review_reason = None
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


@app.command(name="validate")
def validate_words(
    search: str = typer.Option("", "--search", "-s", help="Filter by lemma search"),
    pos: str = typer.Option("", "--pos", "-p", help="Filter by POS"),
    all_words: bool = typer.Option(False, "--all", "-a", help="Validate all words"),
    limit: int = typer.Option(100, "--limit", "-n", help="Maximum words to validate"),
) -> None:
    """Batch validate words with LLM enrichment."""
    if not all_words and not search and not pos:
        error_console.print("[error]Specify --search, --pos, or use --all to validate all words[/]")
        raise typer.Exit(1)

    run_async(_validate_words(search, pos, limit if not all_words else 0))


async def _validate_words(search: str, pos: str, limit: int) -> None:
    """Async implementation of validate command."""
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

        console.print(f"[info]Validating {len(words)} words...[/]\n")

        modified, skipped, deleted, errors = 0, 0, 0, 0

        async def enrich_word(word: Word) -> tuple[Word, EnrichmentResult | Exception]:
            """Enrich a single word, return (word, result) tuple."""
            try:
                enrichment = await enricher.validate_and_enrich(word.lemma, word.pos)
                return (word, enrichment)
            except Exception as e:
                return (word, e)

        # Fire all tasks - semaphore in validate_and_enrich limits concurrency
        tasks = [asyncio.create_task(enrich_word(word)) for word in words]

        with create_progress() as progress:
            task_id = progress.add_task("Validating...", total=len(words))

            for coro in asyncio.as_completed(tasks):
                word, enrich_result = await coro

                if isinstance(enrich_result, Exception):
                    errors += 1
                    progress.update(
                        task_id, advance=1, description=f"[red]Error: {word.display_word}[/]"
                    )
                elif enrich_result.error:
                    errors += 1
                    progress.update(
                        task_id, advance=1, description=f"[red]Error: {word.display_word}[/]"
                    )
                else:
                    try:
                        status = await apply_enrichment_to_word(word, enrich_result, session)
                        if status == "modified":
                            modified += 1
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"[green]Modified: {word.display_word}[/]",
                            )
                        elif status == "deleted":
                            deleted += 1
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"[yellow]Deleted: {word.display_word}[/]",
                            )
                        else:
                            skipped += 1
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"[dim]Skipped: {word.display_word}[/]",
                            )
                    except Exception:
                        errors += 1
                        progress.update(
                            task_id, advance=1, description=f"[red]Error: {word.display_word}[/]"
                        )

                # Commit after each word
                await session.commit()

        console.print()
        console.print(
            f"[success]Complete![/] Modified: {modified}, Skipped: {skipped}, Deleted: {deleted}, Errors: {errors}"
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

        await session.delete(word)
        await session.commit()
        console.print(f"[success]Deleted word: {word.display_word}[/]")
