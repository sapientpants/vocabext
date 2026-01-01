"""Dictionary service for grounding German vocabulary with authoritative sources."""

from app.services.dictionary.base import DictionaryBackend, DictionaryEntry
from app.services.dictionary.service import DictionaryService
from app.services.dictionary.spacy_backend import SpacyBackend, is_model_loaded, preload_model

__all__ = [
    "DictionaryBackend",
    "DictionaryEntry",
    "DictionaryService",
    "SpacyBackend",
    "is_model_loaded",
    "preload_model",
]
