"""Anki sync command."""

from datetime import datetime, timezone

import typer
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.cli.utils.async_runner import run_async
from app.cli.utils.console import console, error_console
from app.cli.utils.progress import create_progress
from app.database import async_session
from app.models import Word
from app.services.anki import AnkiService

app = typer.Typer(
    name="sync",
    help="Anki synchronization commands",
)


@app.command(name="run")
def sync_run() -> None:
    """Sync all unsynced words to Anki."""
    run_async(_sync())


async def _sync() -> None:
    """Async implementation of sync command."""
    anki = AnkiService()

    # Check Anki availability
    console.print("[info]Checking Anki connection...[/]")
    if not await anki.is_available():
        error_console.print("[error]Anki is not running or AnkiConnect is not installed.[/]")
        error_console.print("[dim]Please start Anki and ensure AnkiConnect add-on is installed.[/]")
        raise typer.Exit(1)

    console.print("[success]Connected to Anki[/]")

    # Ensure deck and note type exist
    try:
        await anki.ensure_deck()
        await anki.ensure_note_type()
    except Exception as e:
        error_console.print(f"[error]Failed to setup Anki: {e}[/]")
        raise typer.Exit(1) from None

    async with async_session() as session:
        # Get all words with their versions (needed for needs_sync property)
        stmt = select(Word).options(selectinload(Word.versions))
        result = await session.execute(stmt)
        all_words = result.scalars().all()

        # Delete orphaned Anki notes (notes in Anki that don't exist in database)
        try:
            anki_note_ids = set(await anki.get_all_note_ids())
            db_note_ids = {w.anki_note_id for w in all_words if w.anki_note_id}
            orphaned_ids = list(anki_note_ids - db_note_ids)
            if orphaned_ids:
                await anki.delete_notes(orphaned_ids)
                console.print(f"[dim]Cleaned up {len(orphaned_ids)} orphaned notes from Anki[/]")
        except Exception as e:
            console.print(f"[warning]Failed to clean up orphaned notes: {e}[/]")

        # Filter to only words that need syncing
        words_to_sync = [w for w in all_words if w.needs_sync]
        skipped = len(all_words) - len(words_to_sync)

        if not words_to_sync:
            console.print("\n[success]All words are already synced. No changes needed.[/]")
            return

        console.print(f"\n[info]Syncing {len(words_to_sync)} words to Anki...[/]\n")

        synced = 0
        failed = 0

        with create_progress() as progress:
            task = progress.add_task("Syncing...", total=len(words_to_sync))

            for word in words_to_sync:
                note_id = await anki.sync_word(word)
                if note_id:
                    word.anki_note_id = note_id
                    word.anki_synced_at = datetime.now(timezone.utc)
                    synced += 1
                    progress.update(
                        task, advance=1, description=f"[green]Synced: {word.display_word}[/]"
                    )
                else:
                    failed += 1
                    progress.update(
                        task, advance=1, description=f"[red]Failed: {word.display_word}[/]"
                    )

            await session.commit()

        console.print()
        if failed:
            console.print(
                f"[warning]Synced {synced} words to Anki. {failed} failed. {skipped} already up-to-date.[/]"
            )
        else:
            console.print(f"[success]Successfully synced {synced} words to Anki.[/]")
            if skipped:
                console.print(f"[dim]{skipped} already up-to-date.[/]")
