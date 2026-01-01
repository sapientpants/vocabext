"""spaCy-based dictionary backend for local lemma validation."""

import logging

import spacy
from spacy.language import Language

from app.services.dictionary.base import DictionaryBackend, DictionaryEntry

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model
_nlp: Language | None = None


def _get_nlp() -> Language:
    """Get or load the spaCy German model."""
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy German model...")
        _nlp = spacy.load("de_core_news_lg")
        logger.info("spaCy model loaded")
    return _nlp


def is_model_loaded() -> bool:
    """Check if the spaCy model is already loaded."""
    return _nlp is not None


def preload_model() -> None:
    """Pre-load the spaCy model (call before intensive operations)."""
    _get_nlp()


class SpacyBackend(DictionaryBackend):
    """Local dictionary validation using spaCy's vocabulary."""

    def __init__(self, nlp: Language | None = None) -> None:
        self._nlp = nlp

    @property
    def nlp(self) -> Language:
        """Get the spaCy model, loading if needed."""
        if self._nlp is None:
            self._nlp = _get_nlp()
        return self._nlp

    @property
    def name(self) -> str:
        return "spacy"

    def _is_known(self, word: str) -> bool:
        """Check if a word is in spaCy's vocabulary."""
        lexeme = self.nlp.vocab[word]
        return not lexeme.is_oov

    async def lookup(self, word: str, pos: str | None = None) -> DictionaryEntry | None:
        """
        Look up a word in spaCy vocabulary.

        Note: spaCy vocabulary only validates existence, doesn't provide
        definitions, gender, frequency, etc.
        """
        if not self._is_known(word):
            # Try lowercase
            if self._is_known(word.lower()):
                return DictionaryEntry(
                    lemma=word.lower(),
                    lemma_validated=True,
                    pos=pos,
                    source="spacy",
                )
            return None

        return DictionaryEntry(
            lemma=word,
            lemma_validated=True,
            pos=pos,
            source="spacy",
        )

    async def validate_lemma(self, lemma: str, pos: str | None = None) -> tuple[bool, str | None]:
        """
        Validate that a lemma exists in spaCy vocabulary.

        Returns:
            Tuple of (is_valid, corrected_lemma)
        """
        # Check exact match
        if self._is_known(lemma):
            return True, None

        # Try lowercase
        lower = lemma.lower()
        if lower != lemma and self._is_known(lower):
            return True, lower

        # Try title case for nouns
        if pos == "NOUN":
            title = lemma.title()
            if title != lemma and self._is_known(title):
                return True, title

        # Unknown word - still return True to avoid rejecting valid German words
        # that aren't in spaCy's vocabulary (proper nouns, compounds, etc.)
        logger.debug(f"Word '{lemma}' not in spaCy vocabulary, assuming valid")
        return True, None
