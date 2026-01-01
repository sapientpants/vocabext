"""Main CLI application entry point."""

import typer

from app.cli.commands import process, status, sync, vocabulary
from app.cli.utils.async_runner import run_async
from app.cli.utils.console import error_console
from app.config import settings
from app.database import init_db

app = typer.Typer(
    name="vocabext",
    help="German vocabulary extraction tool with Anki sync",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def startup() -> None:
    """Initialize application on startup."""
    # Ensure directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database
    try:
        run_async(init_db())
    except Exception as e:
        error_console.print(f"[error]Failed to initialize database: {e}[/]")
        raise typer.Exit(1) from None


# Register status command
app.command(name="status", help="Show vocabulary and sync status")(status.status)

# Register vocabulary subcommands
app.add_typer(vocabulary.app, name="vocab")


# Register process command
app.command(name="process", help="Process a document file for vocabulary extraction")(
    process.process_file
)

# Register sync command
app.command(name="sync", help="Sync all unsynced words to Anki")(sync.sync_run)


if __name__ == "__main__":
    app()
