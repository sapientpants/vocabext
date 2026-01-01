"""Base classes and dataclasses for dictionary service."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DictionaryEntry:
    """Unified dictionary lookup result."""

    lemma: str
    lemma_validated: bool = False  # True if dictionary confirmed this lemma
    pos: str | None = None  # NOUN, VERB, ADJ, ADV, ADP
    gender: str | None = None  # der/die/das for nouns
    definitions: list[str] = field(default_factory=list)  # German definitions
    synonyms: list[str] = field(default_factory=list)
    frequency: float | None = None  # 0-6 scale (DWDS)
    ipa: str | None = None  # IPA pronunciation
    source: str = ""  # "dwds", "openthesaurus"
    url: str | None = None  # Link to dictionary entry


class DictionaryBackend(ABC):
    """Abstract base class for dictionary backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this dictionary backend."""
        ...  # pragma: no cover

    @abstractmethod
    async def lookup(self, word: str, pos: str | None = None) -> DictionaryEntry | None:
        """
        Look up a word and return dictionary entry or None if not found.

        Args:
            word: The word to look up
            pos: Optional part of speech to filter results (NOUN, VERB, ADJ, ADV, ADP)

        Returns:
            DictionaryEntry with available data, or None if word not found
        """
        ...  # pragma: no cover

    @abstractmethod
    async def validate_lemma(self, lemma: str, pos: str | None = None) -> tuple[bool, str | None]:
        """
        Validate that a lemma exists in the dictionary.

        Args:
            lemma: The lemma to validate
            pos: Optional part of speech

        Returns:
            Tuple of (is_valid, corrected_lemma)
            - is_valid: True if the lemma exists
            - corrected_lemma: The correct form if different, or None
        """
        ...  # pragma: no cover
