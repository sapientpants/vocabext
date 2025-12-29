"""Extraction review commands."""

import json

import typer
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cli.utils.async_runner import run_async
from app.cli.utils.console import console, error_console
from app.cli.utils.progress import create_progress
from app.database import async_session
from app.models import Document, Extraction, Word

app = typer.Typer(
    name="review",
    help="Extraction review commands",
)


async def _find_or_create_word(
    session: AsyncSession,
    extraction: Extraction,
) -> Word:
    """Find existing word or create new one from extraction."""
    # Check for existing word
    stmt = select(Word).where(
        Word.lemma == extraction.lemma,
        Word.pos == extraction.pos,
    )
    result = await session.execute(stmt)
    existing_word: Word | None = result.scalar_one_or_none()

    if existing_word:
        return existing_word

    # Create new word
    word = Word(
        lemma=extraction.lemma,
        pos=extraction.pos,
        gender=extraction.gender,
        plural=extraction.plural,
        preterite=extraction.preterite,
        past_participle=extraction.past_participle,
        auxiliary=extraction.auxiliary,
        translations=extraction.translations,
    )
    session.add(word)
    await session.flush()
    return word


async def _update_document_status(session: AsyncSession, document_id: int) -> None:
    """Update document status based on remaining pending extractions."""
    pending_count = await session.execute(
        select(func.count(Extraction.id)).where(
            Extraction.document_id == document_id,
            Extraction.status == "pending",
        )
    )
    if pending_count.scalar() == 0:
        document = await session.get(Document, document_id)
        if document:
            document.status = "reviewed"


@app.command(name="list")
def list_extractions(
    document_id: int | None = typer.Option(
        None, "--document-id", "-d", help="Filter by document ID"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum extractions to show"),
) -> None:
    """List pending extractions."""
    run_async(_list_extractions(document_id, limit))


async def _list_extractions(document_id: int | None, limit: int) -> None:
    """Async implementation of list command."""
    async with async_session() as session:
        stmt = select(Extraction).where(Extraction.status == "pending")
        if document_id:
            stmt = stmt.where(Extraction.document_id == document_id)
        stmt = stmt.order_by(Extraction.id).limit(limit)

        result = await session.execute(stmt)
        extractions = result.scalars().all()

        if not extractions:
            console.print("[dim]No pending extractions found.[/]")
            return

        table = Table(title=f"Pending Extractions ({len(extractions)})")
        table.add_column("ID", style="dim", justify="right")
        table.add_column("Doc", style="dim", justify="right")
        table.add_column("Word", style="word")
        table.add_column("POS", style="pos")
        table.add_column("Grammar")
        table.add_column("Translations")

        for ext in extractions:
            trans_list = ext.translations_list[:2]
            trans_str = ", ".join(trans_list) if trans_list else "[dim]None[/]"

            table.add_row(
                str(ext.id),
                str(ext.document_id),
                ext.display_word,
                ext.pos,
                ext.grammar_info or "",
                trans_str,
            )

        console.print(table)


@app.command(name="interactive")
def interactive_review(
    document_id: int | None = typer.Option(
        None, "--document-id", "-d", help="Filter by document ID"
    ),
) -> None:
    """Interactively review pending extractions one by one."""
    run_async(_interactive_review(document_id))


async def _interactive_review(document_id: int | None) -> None:
    """Async implementation of interactive review."""
    async with async_session() as session:
        stmt = select(Extraction).where(Extraction.status == "pending")
        if document_id:
            stmt = stmt.where(Extraction.document_id == document_id)
        stmt = stmt.order_by(Extraction.id)

        result = await session.execute(stmt)
        extractions = list(result.scalars().all())

        if not extractions:
            console.print("[dim]No pending extractions to review.[/]")
            return

        console.print(f"\n[bold]Reviewing {len(extractions)} extractions[/]")
        console.print("[dim]Commands: (a)ccept, (r)eject, (e)dit, (s)kip, (q)uit[/]\n")

        accepted = 0
        rejected = 0
        skipped = 0

        for i, extraction in enumerate(extractions, 1):
            # Display extraction info
            panel_content = f"""[bold]Word:[/] {extraction.display_word}
[bold]POS:[/] {extraction.pos}
[bold]Grammar:[/] {extraction.grammar_info or 'None'}
[bold]Translations:[/] {', '.join(extraction.translations_list) or 'None'}
[bold]Context:[/] [dim]{extraction.context_sentence or 'None'}[/]"""

            console.print(
                Panel(
                    panel_content,
                    title=f"[{i}/{len(extractions)}] Extraction #{extraction.id}",
                    border_style="blue",
                )
            )

            # Get user action
            action = Prompt.ask(
                "Action",
                choices=["a", "r", "e", "s", "q"],
                default="a",
            )

            if action == "q":
                console.print("\n[dim]Review session ended.[/]")
                break
            elif action == "s":
                skipped += 1
                continue
            elif action == "a":
                # Accept extraction
                word = await _find_or_create_word(session, extraction)
                extraction.word_id = word.id
                extraction.status = "accepted"
                await _update_document_status(session, extraction.document_id)
                await session.commit()
                accepted += 1
                console.print("[green]Accepted[/]\n")
            elif action == "r":
                # Reject extraction
                extraction.status = "rejected"
                await _update_document_status(session, extraction.document_id)
                await session.commit()
                rejected += 1
                console.print("[yellow]Rejected[/]\n")
            elif action == "e":
                # Edit extraction
                console.print("[dim]Editing...[/]")

                new_lemma = Prompt.ask("Lemma", default=extraction.lemma)
                new_translations = Prompt.ask(
                    "Translations (comma-separated)",
                    default=", ".join(extraction.translations_list),
                )

                if extraction.pos == "NOUN":
                    new_gender = Prompt.ask("Gender (der/die/das)", default=extraction.gender or "")
                    new_plural = Prompt.ask("Plural", default=extraction.plural or "")
                    extraction.gender = new_gender or None
                    extraction.plural = new_plural or None
                elif extraction.pos == "VERB":
                    new_preterite = Prompt.ask("Preterite", default=extraction.preterite or "")
                    new_pp = Prompt.ask("Past participle", default=extraction.past_participle or "")
                    new_aux = Prompt.ask("Auxiliary", default=extraction.auxiliary or "haben")
                    extraction.preterite = new_preterite or None
                    extraction.past_participle = new_pp or None
                    extraction.auxiliary = new_aux or None

                extraction.lemma = new_lemma
                trans_list = [t.strip() for t in new_translations.split(",") if t.strip()]
                extraction.translations = json.dumps(trans_list) if trans_list else None

                # Ask whether to accept or continue editing
                if Confirm.ask("Accept this extraction?"):
                    word = await _find_or_create_word(session, extraction)
                    extraction.word_id = word.id
                    extraction.status = "accepted"
                    await _update_document_status(session, extraction.document_id)
                    accepted += 1
                    console.print("[green]Accepted[/]\n")

                await session.commit()

        console.print(
            f"\n[success]Review complete![/] Accepted: {accepted}, Rejected: {rejected}, Skipped: {skipped}"
        )


@app.command(name="accept-all")
def accept_all(
    document_id: int = typer.Option(..., "--document-id", "-d", help="Document ID"),
) -> None:
    """Accept all pending extractions for a document."""
    run_async(_accept_all(document_id))


async def _accept_all(document_id: int) -> None:
    """Async implementation of accept-all command."""
    async with async_session() as session:
        # Verify document exists
        document = await session.get(Document, document_id)
        if not document:
            error_console.print(f"[error]Document {document_id} not found[/]")
            raise typer.Exit(1)

        # Get pending extractions
        stmt = select(Extraction).where(
            Extraction.document_id == document_id,
            Extraction.status == "pending",
        )
        result = await session.execute(stmt)
        extractions = list(result.scalars().all())

        if not extractions:
            console.print("[dim]No pending extractions for this document.[/]")
            return

        console.print(f"\n[info]Accepting {len(extractions)} extractions...[/]\n")

        accepted = 0
        with create_progress() as progress:
            task = progress.add_task("Accepting...", total=len(extractions))

            for extraction in extractions:
                word = await _find_or_create_word(session, extraction)
                extraction.word_id = word.id
                extraction.status = "accepted"
                accepted += 1
                progress.update(
                    task, advance=1, description=f"[green]Accepted: {extraction.lemma}[/]"
                )
                await session.commit()

        # Update document status
        document.status = "reviewed"
        await session.commit()

        console.print(f"\n[success]Accepted {accepted} extractions![/]")


@app.command(name="reject-all")
def reject_all(
    document_id: int = typer.Option(..., "--document-id", "-d", help="Document ID"),
) -> None:
    """Reject all pending extractions for a document."""
    run_async(_reject_all(document_id))


async def _reject_all(document_id: int) -> None:
    """Async implementation of reject-all command."""
    async with async_session() as session:
        # Verify document exists
        document = await session.get(Document, document_id)
        if not document:
            error_console.print(f"[error]Document {document_id} not found[/]")
            raise typer.Exit(1)

        # Get pending extractions
        stmt = select(Extraction).where(
            Extraction.document_id == document_id,
            Extraction.status == "pending",
        )
        result = await session.execute(stmt)
        extractions = list(result.scalars().all())

        if not extractions:
            console.print("[dim]No pending extractions for this document.[/]")
            return

        if not Confirm.ask(f"Reject all {len(extractions)} pending extractions?"):
            console.print("[dim]Cancelled.[/]")
            return

        console.print(f"\n[info]Rejecting {len(extractions)} extractions...[/]\n")

        rejected = 0
        with create_progress() as progress:
            task = progress.add_task("Rejecting...", total=len(extractions))

            for extraction in extractions:
                extraction.status = "rejected"
                rejected += 1
                progress.update(task, advance=1)
                await session.commit()

        # Update document status
        document.status = "reviewed"
        await session.commit()

        console.print(f"\n[success]Rejected {rejected} extractions.[/]")
