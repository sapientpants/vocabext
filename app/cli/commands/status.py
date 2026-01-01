"""Status command for displaying application statistics."""

from rich.panel import Panel
from rich.table import Table
from sqlalchemy import func, select

from app.cli.utils.async_runner import run_async
from app.cli.utils.console import console
from app.database import async_session
from app.models import Word
from app.services.anki import AnkiService


def status() -> None:
    """Show vocabulary and sync status."""
    run_async(_status())


async def _status() -> None:
    """Async implementation of status command."""
    anki = AnkiService()
    anki_stats = await anki.get_sync_stats()

    async with async_session() as session:
        # Count words
        word_result = await session.execute(select(func.count(Word.id)))
        word_count: int = word_result.scalar() or 0

        # Count synced words
        synced_result = await session.execute(
            select(func.count(Word.id)).where(Word.anki_note_id.isnot(None))
        )
        synced_count: int = synced_result.scalar() or 0

        # Count words needing review
        review_result = await session.execute(
            select(func.count(Word.id)).where(Word.needs_review == True)  # noqa: E712
        )
        needs_review: int = review_result.scalar() or 0

    # Build vocabulary table
    vocab_table = Table(show_header=False, box=None, padding=(0, 2))
    vocab_table.add_column("Label", style="bold")
    vocab_table.add_column("Value", justify="right")

    vocab_table.add_row("Words", str(word_count))
    vocab_table.add_row("Synced to Anki", f"[green]{synced_count}[/]")
    vocab_table.add_row(
        "Unsynced", f"[yellow]{word_count - synced_count}[/]" if word_count > synced_count else "0"
    )
    vocab_table.add_row("Needs Review", f"[yellow]{needs_review}[/]" if needs_review else "0")

    vocab_panel = Panel(vocab_table, title="[bold]Vocabulary[/]", border_style="blue")

    # Build Anki table
    anki_table = Table(show_header=False, box=None, padding=(0, 2))
    anki_table.add_column("Label", style="bold")
    anki_table.add_column("Value", justify="right")

    if anki_stats.get("available"):
        anki_table.add_row("Status", "[green]Connected[/]")
        anki_table.add_row("Deck", anki_stats.get("deck_name", "N/A"))
        anki_table.add_row(
            "Deck Exists", "[green]Yes[/]" if anki_stats.get("deck_exists") else "[yellow]No[/]"
        )
    else:
        anki_table.add_row("Status", "[red]Disconnected[/]")
        if anki_stats.get("error"):
            anki_table.add_row("Error", f"[dim]{anki_stats['error']}[/]")

    anki_panel = Panel(anki_table, title="[bold]Anki[/]", border_style="blue")

    # Display panels
    console.print()
    console.print(vocab_panel)
    console.print(anki_panel)
    console.print()
