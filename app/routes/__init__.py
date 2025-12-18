"""Route handlers for VocabExt."""

from app.routes.documents import router as documents_router
from app.routes.extractions import router as extractions_router
from app.routes.vocabulary import router as vocabulary_router
from app.routes.sync import router as sync_router

__all__ = [
    "documents_router",
    "extractions_router",
    "vocabulary_router",
    "sync_router",
]
