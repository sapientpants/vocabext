"""Migration script to add versioning support to vocabulary words.

Run this script after updating the models but before starting the app:
    python -m scripts.migrate_versioning

This script will:
1. Add the current_version column to the words table (if missing)
2. Create the word_versions table (if missing)
3. Create initial version (v1) for all existing words that don't have versions
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from app.database import async_session, engine, init_db
from app.models import Word, WordVersion


async def add_current_version_column() -> bool:
    """Add current_version column to words table if it doesn't exist."""
    async with engine.begin() as conn:
        # Check if column exists
        result = await conn.execute(text("PRAGMA table_info(words)"))
        columns = [row[1] for row in result.fetchall()]

        if "current_version" not in columns:
            print("Adding current_version column to words table...")
            await conn.execute(
                text("ALTER TABLE words ADD COLUMN current_version INTEGER DEFAULT 1")
            )
            print("  Column added successfully.")
            return True
        else:
            print("  current_version column already exists.")
            return False


async def create_word_versions_table() -> bool:
    """Create word_versions table if it doesn't exist."""
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='word_versions'")
        )
        if not result.fetchone():
            print("Creating word_versions table...")
            # Let SQLAlchemy create it via init_db
            await init_db()
            print("  Table created successfully.")
            return True
        else:
            print("  word_versions table already exists.")
            return False


async def create_initial_versions() -> int:
    """Create version 1 for all existing words that don't have versions."""
    async with async_session() as session:
        # Get all words
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        stmt = select(Word).options(selectinload(Word.versions))
        result = await session.execute(stmt)
        words = result.scalars().all()

        created = 0
        for word in words:
            # Check if word already has versions
            if not word.versions:
                version = WordVersion(
                    word_id=word.id,
                    version_number=1,
                    lemma=word.lemma,
                    pos=word.pos,
                    gender=word.gender,
                    plural=word.plural,
                    preterite=word.preterite,
                    past_participle=word.past_participle,
                    auxiliary=word.auxiliary,
                    translations=word.translations,
                    created_at=word.created_at,  # Use word's original creation time
                )
                session.add(version)
                created += 1

        if created > 0:
            await session.commit()

        return created


async def migrate() -> None:
    """Run all migration steps."""
    print("=" * 50)
    print("Word Versioning Migration")
    print("=" * 50)

    print("\nStep 1: Add current_version column")
    await add_current_version_column()

    print("\nStep 2: Create word_versions table")
    await create_word_versions_table()

    print("\nStep 3: Create initial versions for existing words")
    created = await create_initial_versions()
    if created > 0:
        print(f"  Created initial versions for {created} words.")
    else:
        print("  No words needed initial versions.")

    print("\n" + "=" * 50)
    print("Migration complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(migrate())
