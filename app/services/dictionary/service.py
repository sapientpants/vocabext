"""Dictionary service facade combining multiple backends with caching."""

import asyncio
import logging
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.dictionary.base import DictionaryBackend, DictionaryEntry
from app.services.dictionary.cache import CacheManager
from app.services.dictionary.spacy_backend import SpacyBackend

logger = logging.getLogger(__name__)


class DictionaryService:
    """
    Facade for dictionary lookups with fallback chain and caching.

    The service tries backends in order until one returns a result.
    Results are cached to reduce API calls.
    """

    def __init__(
        self,
        backends: Sequence[DictionaryBackend] | None = None,
        cache_manager: CacheManager | None = None,
        use_cache: bool = True,
    ) -> None:
        """
        Initialize the dictionary service.

        Args:
            backends: List of dictionary backends to use. Defaults to [SpacyBackend()]
            cache_manager: Cache manager instance. Defaults to CacheManager()
            use_cache: Whether to use caching. Set False for testing.
        """
        self.backends = list(backends) if backends else [SpacyBackend()]
        self.cache_manager = cache_manager or CacheManager()
        self.use_cache = use_cache

    async def lookup(
        self,
        word: str,
        pos: str | None = None,
        session: AsyncSession | None = None,
    ) -> DictionaryEntry | None:
        """
        Look up a word using the fallback chain.

        Args:
            word: The word to look up
            pos: Optional part of speech filter
            session: Database session for caching (optional)

        Returns:
            DictionaryEntry if found, None otherwise
        """
        # Try each backend in order
        for backend in self.backends:
            # Check cache first
            if self.use_cache and session:
                cache_hit, cached_entry = await self.cache_manager.get(
                    session, word, pos, backend.name
                )
                if cache_hit:
                    if cached_entry:
                        logger.debug(f"Cache hit for '{word}' from {backend.name}")
                        return cached_entry
                    else:
                        # Cached "not found", skip this backend
                        logger.debug(f"Cache hit (not found) for '{word}' from {backend.name}")
                        continue

            # Query backend
            try:
                entry = await asyncio.wait_for(
                    backend.lookup(word, pos),
                    timeout=10.0,
                )

                # Cache result
                if self.use_cache and session:
                    await self.cache_manager.set(session, word, pos, backend.name, entry)

                if entry:
                    return entry

            except asyncio.TimeoutError:
                logger.warning(f"Timeout looking up '{word}' in {backend.name}")
            except Exception as e:
                logger.warning(f"Error looking up '{word}' in {backend.name}: {e}")

        return None

    async def validate_and_ground_lemma(
        self,
        lemma: str,
        pos: str,
        session: AsyncSession | None = None,
    ) -> tuple[str, bool, str | None]:
        """
        Validate a lemma against dictionaries and return grounded form.

        Args:
            lemma: The lemma to validate
            pos: Part of speech
            session: Database session for caching (optional)

        Returns:
            Tuple of (final_lemma, is_grounded, source)
            - final_lemma: The validated/corrected lemma
            - is_grounded: True if dictionary confirmed the lemma
            - source: Which dictionary validated it (e.g., "dwds")
        """
        for backend in self.backends:
            try:
                is_valid, corrected = await asyncio.wait_for(
                    backend.validate_lemma(lemma, pos),
                    timeout=10.0,
                )

                if is_valid:
                    final_lemma = corrected if corrected else lemma
                    logger.debug(
                        f"Lemma '{lemma}' validated by {backend.name}"
                        + (f" -> '{corrected}'" if corrected else "")
                    )
                    return final_lemma, True, backend.name

            except asyncio.TimeoutError:
                logger.warning(f"Timeout validating '{lemma}' in {backend.name}")
            except Exception as e:
                logger.warning(f"Error validating '{lemma}' in {backend.name}: {e}")

        # No dictionary could validate
        logger.debug(f"Lemma '{lemma}' not found in any dictionary")
        return lemma, False, None

    async def get_enrichment_data(
        self,
        lemma: str,
        pos: str,
        session: AsyncSession | None = None,
    ) -> DictionaryEntry | None:
        """
        Get enrichment data for a word.

        This is a convenience method that looks up a word and returns
        data that can be used to enrich vocabulary entries.

        Args:
            lemma: The lemma to look up
            pos: Part of speech
            session: Database session for caching (optional)

        Returns:
            DictionaryEntry with available data, or None
        """
        return await self.lookup(lemma, pos, session)

    async def close(self) -> None:
        """Close all backend connections."""
        for backend in self.backends:
            close_method = getattr(backend, "close", None)
            if close_method is not None:
                await close_method()
