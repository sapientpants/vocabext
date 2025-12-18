"""Background tasks for document processing."""

from app.tasks.processing import process_document

__all__ = ["process_document"]
