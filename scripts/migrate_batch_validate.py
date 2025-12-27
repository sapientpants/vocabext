"""Migration script to add batch validation columns to words table.

Run this script after updating the models:
    python -m scripts.migrate_batch_validate

This script will:
1. Add the updated_at column to the words table (if missing)
2. Add the needs_review column to the words table (if missing)
3. Add the review_reason column to the words table (if missing)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from app.database import engine


async def add_columns() -> None:
    """Add new columns to words table if they don't exist."""
    async with engine.begin() as conn:
        # Check existing columns
        result = await conn.execute(text("PRAGMA table_info(words)"))
        columns = [row[1] for row in result.fetchall()]

        # Add updated_at column
        if "updated_at" not in columns:
            print("Adding updated_at column to words table...")
            # SQLite doesn't allow non-constant defaults, so add column first
            await conn.execute(text("ALTER TABLE words ADD COLUMN updated_at DATETIME"))
            # Set updated_at to created_at for existing rows
            await conn.execute(text("UPDATE words SET updated_at = created_at"))
            print("  Column added successfully.")
        else:
            print("  updated_at column already exists.")

        # Add needs_review column
        if "needs_review" not in columns:
            print("Adding needs_review column to words table...")
            await conn.execute(text("ALTER TABLE words ADD COLUMN needs_review BOOLEAN DEFAULT 0"))
            print("  Column added successfully.")
        else:
            print("  needs_review column already exists.")

        # Add review_reason column
        if "review_reason" not in columns:
            print("Adding review_reason column to words table...")
            await conn.execute(text("ALTER TABLE words ADD COLUMN review_reason TEXT"))
            print("  Column added successfully.")
        else:
            print("  review_reason column already exists.")


async def migrate() -> None:
    """Run all migration steps."""
    print("=" * 50)
    print("Batch Validation Migration")
    print("=" * 50)

    await add_columns()

    print("\n" + "=" * 50)
    print("Migration complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(migrate())
